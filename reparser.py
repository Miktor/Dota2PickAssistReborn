import logging
import json
import glob
import os

MATCHES_FOLDER = 'D:\matches\matches\matches'
SAVE_FILE = 'packed.json'


def reparse(data):
    try:
        if data['human_players'] is not 10:
            logging.exception('human_players = %d', data['human_players'])
            return

        packed = {'match_id': data['match_id'],
                  'region': data['region'],
                  'start_time': data['start_time'],
                  'version': '{}.{}'.format(data['version'], data['patch']),
                  'radiant_win': data['radiant_win']}

        heroes = []
        for player in data['players']:
            heroes.append({'hero_id': player['hero_id'],
                           'isRadiant': player['isRadiant']})

        packed['heroes'] = heroes

        return packed
    except Exception:
        logging.exception('Failed to pack')


def main():
    try:
        list_of_files = glob.glob(MATCHES_FOLDER + '/*.json')
        latest_file = max(list_of_files, key=os.path.getctime)

        logging.info('Found %u files', len(list_of_files))

        out_data = []
        for filename in list_of_files:
            with open(filename, 'rt') as in_file:
                try:
                    data = json.loads(in_file.read())
                    if data:
                        res = reparse(data)
                        if res:
                            out_data.append(res)

                except json.decoder.JSONDecodeError:
                    pass
                except Exception as e:
                    logging.exception('Failed to read file %s', filename)

        with open(SAVE_FILE, 'wt') as out_file:
            json.dump(out_data, out_file)

    except Exception as e:
        logging.exception('Failed to enumerate files')


if __name__ == '__main__':
    main()
