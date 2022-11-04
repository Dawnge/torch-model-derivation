use tch::{self, nn::{self, Module}};


/// Generic Model class
/// 


pub fn custom_net(vs: &nn::Path, inputs: i64, outputs: i64, hidden_layers: i64, layer_size: i64) -> impl Module {
    let mut net = nn::seq();

    // Input Layer
    net = net.add(nn::linear(vs, inputs, layer_size, Default::default()));
    net = net.add(nn::func(|xs| xs.leaky_relu()));

    for _ in 0..hidden_layers {
        net = net.add(nn::linear(vs, layer_size, layer_size, Default::default()));
        net = net.add(nn::func(|xs| xs.leaky_relu()));
    }
    
    // output layer
    net.add(nn::linear(vs, layer_size, outputs, Default::default()))
}





pub struct OneCycleLr {
    last_step: i64,
    min_lr: f64,
    max_lr: f64,
    min_momentum: f64,
    max_momentum: f64,
    num_steps: i64,
    num_cycle_steps: i64,
    final_lr: f64,
    pub last_lr: f64,
    pub last_momentum: f64,
}

impl OneCycleLr {
    pub fn new(num_steps: i64, lr_range: (f64,f64), momentum_range: (f64,f64), annihilation_frac: f64, reduce_factor: f64) -> Self {
        if lr_range.0 > lr_range.1 {
            panic!("lr_range min > lr_range max");
        }
        if lr_range.0 > lr_range.1 {
            panic!("momentum_range min > momentum_range max");
        }
        let mut oclr = Self {
            last_step: -1,
            num_steps,
            min_lr: lr_range.0,
            max_lr: lr_range.1,
            min_momentum: momentum_range.0,
            max_momentum: momentum_range.1,
            num_cycle_steps: (num_steps as f64 * (1.-annihilation_frac)) as i64,
            final_lr: lr_range.0 * reduce_factor,
            last_lr: 0.,
            last_momentum: 0.,
        };

        oclr.step();
        oclr
    }

    pub fn step(&mut self) -> (f64,f64) {
        let current_step = self.last_step + 1;
        self.last_step = current_step;

        let mut lr = -1.;
        let mut momentum = -1.;

        if current_step <= self.num_cycle_steps / 2{
            let scale = current_step as f64 / (self.num_cycle_steps as f64 / 2. ); // should be floor division
            lr = self.min_lr + (self.max_lr - self.min_lr) * scale;
            momentum = self.max_momentum - (self.max_momentum - self.min_momentum) * scale;
        } else if current_step <= self.num_cycle_steps {
            let scale = (current_step as f64 - self.num_cycle_steps as f64 / 2.) / (self.num_cycle_steps as f64 - self.num_cycle_steps as f64 / 2.);
            lr = self.max_lr - (self.max_lr - self.min_lr) * scale as f64;
            momentum = self.min_momentum + (self.max_momentum - self.min_momentum) * scale as f64;
        } else if current_step <= self.num_steps {
            let scale = (current_step - self.num_cycle_steps) as f64 / (self.num_steps - self.num_cycle_steps) as f64;
            lr = self.min_lr - (self.min_lr - self.final_lr) * scale;
            momentum = 0.;
        } else {
            println!("WARNING: scheduler doing nothing");
        }
        self.last_lr = lr;
        self.last_momentum = momentum;
        (lr, momentum)
    }
}