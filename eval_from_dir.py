from eval_metrics import compute_metrics

metric = compute_metrics(200, "./eval_output/eval_canny_base", "canny")
metric.compute()