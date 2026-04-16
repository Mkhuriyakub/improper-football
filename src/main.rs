use ball::football::Football;
use physics::math::Vec3;

fn main() {
    println!("improper-football — physics engine starting");

    // Demo: simulate a spiral throw and print trajectory
    let mut ball = Football::new();

    // Typical NFL throw: ~20 m/s at 35° upward angle, 600 RPM spiral
    let speed = 20.0_f64;
    let angle = 35.0_f64.to_radians();
    let velocity = Vec3::new(speed * angle.cos(), speed * angle.sin(), 0.0);
    ball.throw_spiral(velocity, 600.0);

    println!("Throw: {:.1} m/s at {:.0}°, 600 RPM spiral", speed, 35.0);
    println!("{:<8} {:<10} {:<10} {:<10}", "t (s)", "x (m)", "y (m)", "z (m)");

    let dt = 0.01;
    let mut t = 0.0_f64;

    while ball.body.position.y >= 0.0 || t < dt {
        let p = ball.body.position;
        if (t * 100.0).round() as i32 % 10 == 0 {
            println!("{:<8.2} {:<10.2} {:<10.2} {:<10.2}", t, p.x, p.y, p.z);
        }
        ball.step(dt);
        t += dt;
        if t > 10.0 { break; }
    }

    println!("\nBall landed at x = {:.2} m", ball.body.position.x);
}
