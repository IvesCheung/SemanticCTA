class NullMlflowLogger:
    def log_param(self, *args, **kwargs):
        return None

    def log_params(self, *args, **kwargs):
        return None

    def log_metric(self, *args, **kwargs):
        return None

    def log_metrics(self, *args, **kwargs):
        return None

    def set_tag(self, *args, **kwargs):
        return None


def get_mlflow_logger(enabled=False):
    if not enabled:
        return NullMlflowLogger()

    import mlflow

    return mlflow
