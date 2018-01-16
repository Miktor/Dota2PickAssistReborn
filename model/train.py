import json

PACKED_FILE = 'packed.json'


def read(path=PACKED_FILE):
    with open(path, 'r') as f:
        return json.loads(f.read())


def main():
    data = read()
    # TODO: do something


if __name__ == '__main__':
    main()
