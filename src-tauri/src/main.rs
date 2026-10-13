// Hide the console window in release builds on Windows.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

fn main() {
    // `mahdar.exe --watchdog` runs the watchdog (no window) instead of the app.
    let args: Vec<String> = std::env::args().collect();
    if args.iter().any(|a| a == "--watchdog") {
        mahdar_lib::watchdog::run(args.iter().any(|a| a == "--at-login"));
        return;
    }
    mahdar_lib::run()
}
