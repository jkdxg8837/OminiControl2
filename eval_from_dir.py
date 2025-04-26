from eval_metrics import compute_metrics

metric = compute_metrics(200, "./eval_canny", "canny")
metric.compute()