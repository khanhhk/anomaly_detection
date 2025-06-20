from adbench.run import RunPipeline
# pipeline = RunPipeline(
#     dataset_path='adbench/datasets/Classical/2_annthyroid.npz',
#     suffix='ADBench',
#     mode='rla',
#     parallel='unsupervise',
#     seed=42
# )
# results = pipeline.run()

# pipeline = RunPipeline(
#     dataset_path='adbench/datasets/Classical/2_annthyroid.npz',
#     suffix='ADBench',
#     mode='rla',
#     parallel='semi-supervise',
#     seed=42
# )
# results = pipeline.run()

pipeline = RunPipeline(
    dataset_path='adbench/datasets/Classical/2_annthyroid.npz',
    suffix='ADBench',
    mode='rla',
    parallel='supervise',
    seed=42
)
results = pipeline.run()
