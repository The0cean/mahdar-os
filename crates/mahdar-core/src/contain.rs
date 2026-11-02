//! Running a writer's command-line program (Cursor, Claude Code) so it can't run away.
//!
//! On Windows the program and everything it starts go into one job object:
//! - at most [`MAX_AT_ONCE`] of them run at the same time,
//! - if they keep starting new programs ([`MAX_STARTED`] in one call), all are stopped,
//! - whatever is still running when the call ends (or times out) is closed with it.
//!
//! So a misbehaving CLI can't fill the screen with terminal windows. Each call logs how many
//! programs it started.

use std::io::{Read, Write};
use std::process::{Command, Output, Stdio};
use std::time::{Duration, Instant};

use anyhow::{bail, Context, Result};

/// Programs of one call that may run at the same time.
pub const MAX_AT_ONCE: u32 = 40;
/// Programs one call may start in total before it is stopped.
pub const MAX_STARTED: u32 = 300;

/// Run `cmd` with `stdin` as its input and wait for it, at most `limit`. `name` is for
/// messages ("Cursor"). stdout and stderr are captured.
// Dropping the job closes it (and its programs) on Windows; elsewhere there is nothing to drop.
#[cfg_attr(not(windows), allow(clippy::drop_non_drop))]
pub fn output(cmd: &mut Command, name: &str, stdin: Option<String>, limit: Duration) -> Result<Output> {
    cmd.stdin(if stdin.is_some() { Stdio::piped() } else { Stdio::null() }).stdout(Stdio::piped()).stderr(Stdio::piped());
    let mut child = cmd.spawn().with_context(|| format!("could not start {name}"))?;
    let job = Job::hold(&child);
    if job.is_none() && cfg!(windows) {
        crate::log::warn("writer", format!("{name}: could not put it in a job; it runs unguarded"));
    }
    let writer = child.stdin.take().zip(stdin).map(|(mut pipe, text)| {
        std::thread::spawn(move || {
            let _ = pipe.write_all(text.as_bytes());
        })
    });
    let read = |mut r: Box<dyn Read + Send>| {
        std::thread::spawn(move || {
            let mut buf = Vec::new();
            let _ = r.read_to_end(&mut buf);
            buf
        })
    };
    let out = child.stdout.take().map(|s| read(Box::new(s)));
    let err = child.stderr.take().map(|s| read(Box::new(s)));

    let t0 = Instant::now();
    let status = loop {
        if let Some(status) = child.try_wait()? {
            break status;
        }
        let started = job.as_ref().map(|j| j.started()).unwrap_or(0);
        if started > MAX_STARTED {
            drop(job); // closes every program it started
            let _ = child.kill();
            let _ = child.wait();
            crate::log::error("writer", format!("{name} kept starting new programs ({started} in {} s); stopped it", t0.elapsed().as_secs()));
            bail!("{name} kept starting new programs ({started}), so Mahdar stopped it. Try another engine in Settings, and send the log (Monitor → OPEN LOG FOLDER) so this can be fixed.");
        }
        if t0.elapsed() > limit {
            drop(job);
            let _ = child.kill();
            let _ = child.wait();
            bail!("{name} didn't answer within {} minutes, so Mahdar stopped it", limit.as_secs() / 60);
        }
        std::thread::sleep(Duration::from_millis(100));
    };
    if let Some(j) = &job {
        let (started, left) = (j.started(), j.running());
        crate::log::info(
            "writer",
            format!("{name} started {started} program{}{}", if started == 1 { "" } else { "s" }, if left > 0 { format!("; closing {left} it left running") } else { String::new() }),
        );
    }
    drop(job); // closes anything it left running, so its pipes close too
    if let Some(w) = writer {
        let _ = w.join();
    }
    let stdout = out.map(|t| t.join().unwrap_or_default()).unwrap_or_default();
    let stderr = err.map(|t| t.join().unwrap_or_default()).unwrap_or_default();
    Ok(Output { status, stdout, stderr })
}

#[cfg(windows)]
struct Job(windows_sys::Win32::Foundation::HANDLE);

// The handle is only used to query and close the job.
#[cfg(windows)]
unsafe impl Send for Job {}

#[cfg(windows)]
impl Job {
    /// A new job holding `child`: limited to MAX_AT_ONCE processes, all killed when dropped.
    fn hold(child: &std::process::Child) -> Option<Job> {
        use std::os::windows::io::AsRawHandle;
        use windows_sys::Win32::Foundation::CloseHandle;
        use windows_sys::Win32::System::JobObjects::*;
        unsafe {
            let job = CreateJobObjectW(std::ptr::null(), std::ptr::null());
            if job.is_null() {
                return None;
            }
            let mut info: JOBOBJECT_EXTENDED_LIMIT_INFORMATION = std::mem::zeroed();
            info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE | JOB_OBJECT_LIMIT_ACTIVE_PROCESS;
            info.BasicLimitInformation.ActiveProcessLimit = MAX_AT_ONCE;
            let set = SetInformationJobObject(
                job,
                JobObjectExtendedLimitInformation,
                &info as *const _ as *const core::ffi::c_void,
                std::mem::size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>() as u32,
            );
            if set == 0 || AssignProcessToJobObject(job, child.as_raw_handle() as _) == 0 {
                CloseHandle(job);
                return None;
            }
            Some(Job(job))
        }
    }

    fn accounting(&self) -> windows_sys::Win32::System::JobObjects::JOBOBJECT_BASIC_ACCOUNTING_INFORMATION {
        use windows_sys::Win32::System::JobObjects::*;
        unsafe {
            let mut info: JOBOBJECT_BASIC_ACCOUNTING_INFORMATION = std::mem::zeroed();
            QueryInformationJobObject(
                self.0,
                JobObjectBasicAccountingInformation,
                &mut info as *mut _ as *mut core::ffi::c_void,
                std::mem::size_of::<JOBOBJECT_BASIC_ACCOUNTING_INFORMATION>() as u32,
                std::ptr::null_mut(),
            );
            info
        }
    }

    /// Programs started so far, the first one included.
    fn started(&self) -> u32 {
        self.accounting().TotalProcesses
    }

    fn running(&self) -> u32 {
        self.accounting().ActiveProcesses
    }
}

#[cfg(windows)]
impl Drop for Job {
    fn drop(&mut self) {
        unsafe {
            windows_sys::Win32::Foundation::CloseHandle(self.0);
        }
    }
}

/// Elsewhere there is no job: the program runs as is, with the time limit.
#[cfg(not(windows))]
struct Job;

#[cfg(not(windows))]
impl Job {
    fn hold(_: &std::process::Child) -> Option<Job> {
        None
    }
    fn started(&self) -> u32 {
        0
    }
    fn running(&self) -> u32 {
        0
    }
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;

    #[test]
    fn captures_output_and_input() {
        let out = output(Command::new("sh").args(["-c", "tr a-z A-Z; echo oops >&2"]), "sh", Some("hello".into()), Duration::from_secs(10)).unwrap();
        assert!(out.status.success());
        assert_eq!(String::from_utf8_lossy(&out.stdout), "HELLO");
        assert_eq!(String::from_utf8_lossy(&out.stderr).trim(), "oops");
    }

    #[test]
    fn stops_at_the_time_limit() {
        let t0 = Instant::now();
        let e = output(&mut Command::new("sleep").arg("30"), "sleep", None, Duration::from_millis(300)).unwrap_err();
        assert!(t0.elapsed() < Duration::from_secs(5));
        assert!(e.to_string().contains("didn't answer"), "{e}");
    }
}
