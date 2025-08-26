import os 
import numpy as np 

def run_mcmc_for_all_datasets(experiment_path, 
                              num_chains=3, 
                              num_samples=6000, 
                              dt=0.05, 
                              lambd=0.8, 
                              obs_var=0.001, 
                              rb=2, 
                              prior_sigma_min=0.0, 
                              prior_sigma_max=1.0, 
                              prior_alpha_min=0.0, 
                              prior_alpha_max=0.03, 
                              proposal_sigma_tau=0.2, 
                              proposal_alpha_tau=0.015, 
                              use_wandb=True):
    """Run MCMC for all datasets in the experiment path"""

    
    # Get all dataset folders
    dataset_folders = glob.glob(f"{experiment_path}/seed=*")
    
    print(f"Found {len(dataset_folders)} datasets in {experiment_path}")
    
    # Loop through each dataset folder
    for dataset_folder in dataset_folders:
        # Extract the folder name
        folder_name = os.path.basename(dataset_folder)
        print(f"\nProcessing dataset: {folder_name}")
        
        # Check if the required file exists
        data_file = f"{dataset_folder}/procrustes_aligned.csv"
        if not os.path.exists(data_file):
            print(f"  Skipping: {data_file} not found")
            continue
        
        # Generate a random seed for this batch of chains
        seed_start = np.random.randint(0, 1000_000_000)
        
        # Set up output path
        output_path = f"{dataset_folder}/mcmc_seed={seed_start}_N={num_samples}"
        
        print(f"  Starting {num_chains} MCMC chains with seed {seed_start}")
        
        # Start MCMC chains in screen sessions
        screen_sessions = run_mcmc_in_screens_cpu(
            num_chains=num_chains,
            script_path="run_mcmc.py",
            seed_param="--seed_mcmc",
            seed_start=seed_start,
            screen_prefix=f"mcmc_{folder_name}",  # Use unique screen names
            max_concurrent=2,  # Only run 2 chains at a time
            cpu_ids_per_chain=1,  # Use 1 CPU per chain
            script_args={
                "--outputpath": output_path,
                "--phylopath": "../data/chazot_subtree_rounded.nw",
                "--datapath": data_file,
                "--dt": dt,
                "--lambd": lambd,
                "--obs_var": obs_var,
                "--rb": rb,
                "--N": num_samples,
                "--prior_sigma_min": prior_sigma_min,
                "--prior_sigma_max": prior_sigma_max,
                "--prior_alpha_min": prior_alpha_min,
                "--prior_alpha_max": prior_alpha_max,
                "--proposal_sigma_tau": proposal_sigma_tau,
                "--proposal_alpha_tau": proposal_alpha_tau,
                "--use_wandb": True
            }
        )
        
        print(f"  Started chains for {folder_name}. Screen sessions: {', '.join(screen_sessions)}")
        
        # Optional: Add a delay between datasets to avoid overloading the system
        time.sleep(200)
    
    print(f"\nMCMC chains started for all datasets in {experiment_path}")



def run_mcmc_in_screens_cpu(num_chains, script_path="run_mcmc.py", screen_prefix="mcmc_chain", 
                       seed_start=42, seed_param="--seed_mcmc", script_args=None,
                       max_concurrent=2, cpu_ids_per_chain=1):
    """
    Run multiple MCMC chains with CPU resource management.
    
    Args:
        max_concurrent: Maximum number of chains to run concurrently
        cpu_ids_per_chain: How many CPU cores to use per chain (can be a list or int)
    """
    screen_names = []
    
    # Convert script_args dictionary to command-line string
    args_str = ""
    if script_args:
        for key, value in script_args.items():
            args_str += f" {key} {value}"
    
    # Number of available CPUs
    total_cpus = os.cpu_count()
    print(f"Server has {total_cpus} CPUs available")
    
    # Map chains to specific CPU cores in a round-robin fashion
    for i in range(num_chains):
        # Wait if we've reached the concurrent limit
        if i >= max_concurrent and i % max_concurrent == 0:
            print(f"Waiting 60 seconds before launching next batch...")
            time.sleep(60)
            
        seed = seed_start + i
        screen_name = f"{screen_prefix}_{i+1}"
        screen_names.append(screen_name)
        
        # Determine which CPU(s) to use for this chain
        if isinstance(cpu_ids_per_chain, int):
            # Assign sequential CPUs to each chain, wrap around if needed
            start_cpu = (i * cpu_ids_per_chain) % total_cpus
            cpu_list = list(range(start_cpu, min(start_cpu + cpu_ids_per_chain, total_cpus)))
            # Handle wrap-around
            if start_cpu + cpu_ids_per_chain > total_cpus:
                cpu_list += list(range(0, (start_cpu + cpu_ids_per_chain) % total_cpus))
            cpu_str = ",".join(map(str, cpu_list))
        else:
            # Use the provided list directly
            cpu_str = str(cpu_ids_per_chain[i % len(cpu_ids_per_chain)])
            
        # Use taskset to pin the process to specific CPU cores
        cmd = (
            f"screen -dmS {screen_name} bash -c '"
            f"taskset -c {cpu_str} python {script_path} {seed_param} {seed}{args_str}; "
            "echo \"Chain complete, press Ctrl+C to exit.\"; "
            "sleep 10'"
        )
        
        print(f"Starting chain {i+1} with seed {seed} on CPU(s) {cpu_str}")
        subprocess.run(cmd, shell=True)
        time.sleep(2)  # Small delay between launches
    
    return screen_names



def run_mcmc_in_screens(num_chains, script_path="run_mcmc.py", screen_prefix="mcmc_chain", 
                         seed_start=42, seed_param="--seed_mcmc", script_args=None):
    """
    Run multiple MCMC chains in separate screen sessions with customizable arguments.
    
    Args:
        num_chains: Number of chains to run
        script_path: Path to the MCMC script
        screen_prefix: Prefix for screen session names
        seed_start: Starting seed value
        seed_param: Parameter name for seed in script
        script_args: Dictionary of additional arguments to pass to the script
                    (e.g., {"--dt": 0.05, "--N": 5000})
    
    Returns:
        List of screen session names
    """
    screen_names = []
    
    # Convert script_args dictionary to command-line string
    args_str = ""
    if script_args:
        for key, value in script_args.items():
            args_str += f" {key} {value}"
    
    for i in range(num_chains):
        seed = seed_start + i
        screen_name = f"{screen_prefix}_{i+1}"
        screen_names.append(screen_name)
        
        # Create the screen session and run the script with all arguments
        cmd = (
            f"screen -dmS {screen_name} bash -c '"
            f"python {script_path} {seed_param} {seed}{args_str}; "
            "echo \"Chain complete, press Ctrl+C to exit.\"; "
            "sleep infinity'"
        )
        
        print(f"Starting chain {i+1} with seed {seed} in screen '{screen_name}'")
        subprocess.run(cmd, shell=True)
        time.sleep(1)  # Small delay to avoid resource contention
    
    print(f"\n{num_chains} MCMC chains started in separate screen sessions.")
    print("To attach to a screen session: screen -r <screen_name>")
    print("To detach from a screen session: Ctrl+A, then D")
    print(f"Screen sessions: {', '.join(screen_names)}")
    
    return screen_names