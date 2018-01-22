from sys import stdout


def progress_bar(current, total):
    barLength = 20  # Modify this to change the length of the progress bar
    status = ""

    progress = float(current) / float(total)

    if progress < 0:
        progress = 0
        status = "Halt...\r\n"
    if progress >= 1:
        progress = 1
        status = "Done...\r\n"
    block = int(round(barLength * progress))
    text = "\rPercent: [{0}] {1:.3f}% {2}".format("=" * block + " " * (barLength - block), progress * 100, status)
    stdout.write(text)
    stdout.flush()