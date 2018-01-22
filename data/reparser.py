import logging
import json
import glob
import os
import operator

from utils.utils import progress_bar

MATCHES_FOLDER = 'data/raw'
SAVE_FILE = 'packed.json'

unparsed = 0
total = 0
leaved = 0
radiant_wins = 0
dire_winds = 0
avg_mmr = 0
avg_duration = 0
heroes_picked_times = {}

ADVANCED_STATS = False


def reparse(data):
    global unparsed
    global total
    global leaved
    global radiant_wins
    global dire_winds
    global avg_mmr
    global avg_duration
    global heroes_picked_times

    try:
        if ADVANCED_STATS and ('version' not in data or data['version'] is None):
            logging.info('unparsed')
            unparsed += 1
            return

        if data['human_players'] is not 10:
            logging.info('human_players = %d', data['human_players'])
            return

        if data['duration'] < 10 * 60:
            logging.info('duration < 10 min')
            return

        radiant_win = data['radiant_win']
        mmr = data['mmr']
        duration = data['duration']
        packed = {
            'match_id': data['match_id'],
            'mmr': mmr,
            'start_time': data['start_time'],
            'duration': duration,
            'radiant_win': radiant_win
        }

        total += 1
        if radiant_win:
            radiant_wins += 1
        else:
            dire_winds += 1

        avg_mmr += mmr
        avg_duration += duration

        heroes = []
        for player in data['players']:

            if player['abandons'] != 0:
                logging.info('leaver')
                leaved += 1
                return

            hero_id = player['hero_id']

            heroes.append({'hero_id': hero_id, 'isRadiant': player['isRadiant']})

            if ADVANCED_STATS:
                heroes.append({
                    'lane_efficiency': player['lane_efficiency'],
                    'lane': player['lane'],
                    'lane_role': player['lane_role'],
                    'is_roaming': player['is_roaming']
                })

            if hero_id in heroes_picked_times:
                heroes_picked_times[hero_id] += 1
            else:
                heroes_picked_times[hero_id] = 1

        packed['heroes'] = heroes

        return packed
    except Exception:
        logging.exception('Failed to pack')


def main():
    try:
        FORMAT = '%(asctime)-15s %(clientip)s %(user)-8s %(message)s'

        # logging.basicConfig(format=FORMAT, level=logging.DEBUG)

        list_of_files = glob.glob(MATCHES_FOLDER + '/*.json')
        latest_file = max(list_of_files, key=os.path.getctime)

        logging.info('Found %d files', len(list_of_files))

        out_data = []
        total = len(list_of_files)

        for i, filename in enumerate(list_of_files):
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
            progress_bar(i, total)

        with open(SAVE_FILE, 'wt') as out_file:
            json.dump(out_data, out_file)

    except Exception as e:
        logging.exception('Failed to enumerate files')


if __name__ == '__main__':
    main()

    print(
        'Stats:\r\n\tunparsed = {}\r\n\ttotal = {}\r\n\tleaved = {}\r\n\tradiant: {}\r\n\tdire: {}\r\n\tavg_duration: {}\r\n\tmmr: {}\r\n\theroes: {}'.
        format(unparsed, total, leaved, radiant_wins, dire_winds, avg_duration / total, avg_mmr / total,
               sorted(heroes_picked_times.items(), key=operator.itemgetter(1), reverse=True)))
