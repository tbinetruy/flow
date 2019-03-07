import os

MAX_BUF = 1024


def write_message():
    myfifo = "/tmp/fifopipe2"
    os.mkfifo(myfifo, 0o666)

    fd = os.open(myfifo, os.O_WRONLY)

    msg=b"This is the string to be reversed"
    os.write(fd, msg)
    os.close(fd)
    os.unlink(myfifo)


def read_message():
    myfifo = "/tmp/fifopipe"
    buf = ""
    fd = os.open(myfifo, os.O_RDONLY)

    while len(buf) == 0:
        buf = os.read(fd, MAX_BUF)

    print("Received:", buf.decode())
    os.close(fd)


read_message()
write_message()
