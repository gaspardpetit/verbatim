def format_milliseconds(ms: float) -> str:
    # Calculate hours, minutes, and seconds
    seconds_total = int(ms // 1000)
    hours = int(seconds_total // 3600)
    minutes = int((seconds_total % 3600) // 60)
    seconds = int(seconds_total) % 60

    return f"{hours:02}:{minutes:02}:{seconds:02}"
