## Summary


Run simulations from a config file, giving the prior range and which parameter to sample
- `CONFIG_sbi = narrow_prior_1_param` (in the `config_sbi.py` file)
- `python sbi_run_simulations.py --config_to_simulate CONFIG_sbi --seed 30 --n_sims 200 --checkpoint_interval 10 --n_cores 3`

Train posteriors
- `python sbi_train_posteriors.py --config_to_train CONFIG_sbi`

Choose a data vector to test for sampling the posterior
- `CONFIG_sample = narrow_prior_1_param` (in the `config_posterior_sampling.py` file)
- `python sbi_sample_posteriors.py --config_to_sample CONFIG_sample`
  