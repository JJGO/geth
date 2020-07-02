
def FlushLog(experiment):

    def flush(experiment, _):
        print("", end="", flush=True)
    return flush
