from adbench.run import RunPipeline
# pipeline = RunPipeline(suffix='ADBench', parallel='unsupervise', realistic_synthetic_mode = None, noise_type=None)
# results = pipeline.run()

# pipeline = RunPipeline(suffix='ADBench', parallel='semi-supervise', realistic_synthetic_mode = None, noise_type=None)
# results = pipeline.run()

pipeline = RunPipeline(suffix='ADBench', parallel='supervise', realistic_synthetic_mode = None, noise_type=None)
results = pipeline.run()