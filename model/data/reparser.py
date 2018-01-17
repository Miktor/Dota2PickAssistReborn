import logging
import json
import glob
import os
import operator

MATCHES_FOLDER = 'raw'
SAVE_FILE = 'packed.json'

unparsed = 0
total = 0
leaved = 0
radiant_wins = 0
dire_winds = 0
avg_mmr = 0
heroes_picked_times = {}


def reparse(data):
    global unparsed
    global total
    global leaved
    global radiant_wins
    global dire_winds
    global avg_mmr
    global heroes_picked_times

    try:
        if 'version' not in data or data['version'] is None:
            logging.info('unparsed')
            unparsed += 1
            return

        if data['human_players'] is not 10:
            logging.info('human_players = %d', data['human_players'])
            return

        if data['duration'] < 10 * 60:
            logging.info('duration < 10 min')
            return

        for player in data['players']:
            if player['leaver_status'] == 1:
                logging.info('leaver')
                leaved += 1
                return

        radiant_win = data['radiant_win']
        mmr = data['mmr']
        packed = {'match_id': data['match_id'],
                  'mmr': mmr,
                  'start_time': data['start_time'],
                  'duration': data['duration'],
                  'radiant_win': radiant_win}

        total += 1
        if radiant_win:
            radiant_wins += 1
        else:
            dire_winds += 1
        avg_mmr += mmr

        heroes = []
        for player in data['players']:
            hero_id = player['hero_id']
            heroes.append({'hero_id': hero_id,
                           'isRadiant': player['isRadiant'],
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

    print('Stats: unparsed = {}, total = {}, leaved = {}, radiant: {}, dire: {}, mmr: {}, heroes: {}'.format(unparsed,
                                                                                                             total,
                                                                                                             leaved,
                                                                                                             radiant_wins,
                                                                                                             dire_winds,
                                                                                                             avg_mmr / total,
                                                                                                             sorted(heroes_picked_times.items(), key=operator.itemgetter(1), reverse=True)))
