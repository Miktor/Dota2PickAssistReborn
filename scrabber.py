import json
import datetime
import os
import sys
import glob
import time
import logging
import logging.config
import urllib.request
import urllib.parse


def query_matches(starting_match_id, min_mmr):
    try:
        sql_query = ("select * from public_matches "
                     "where avg_mmr >= {0} and match_id > {1} "
                     "and lobby_type = 7 and game_mode in (2, 22) "
                     "ORDER BY start_time asc "
                     "limit 100")

        query = "https://api.opendota.com/api/explorer?sql=" + urllib.parse.quote(
            sql_query.format(min_mmr, starting_match_id))

        logging.debug(
            'Quering match from mmr %d, min match id = %d, string = %s', min_mmr,
            starting_match_id, query)

        while True:
            response = urllib.request.urlopen(query)
            if response.code is not 200:
                logging.warning(
                    'Failed to query, return code = %d', response.code)
                time.sleep(5)
                logging.warning('Retrying')
            else:
                break

        data = json.loads(response.read().decode("utf-8"))

        rows = data['rows']
        matches = []
        for row in rows:
            match_id = row['match_id']
            matches.append(match_id)
            logging.debug(
                'match_id = %d, mmr = %d, date = %s', match_id, row['avg_mmr'],
                str(datetime.datetime.fromtimestamp(row['start_time'])))
        return matches
    except Exception as e:
        logging.exception('Failed to query match info')
        return []


def query_match(match_id):
    try:
        query = "https://api.opendota.com/api/matches/{}".format(match_id)
        logging.debug('Quering match %d, string = %s', match_id, query)

        while True:
            response = urllib.request.urlopen(query)
            if response.code is not 200:
                logging.warning(
                    'Failed to query, return code = %d', response.code)
            else:
                break

        data = json.loads(response.read().decode("utf-8"))
        return data
    except Exception as e:
        logging.exception('Failed to query matches')


def main():
    first_match = 3534655441
    save_folder = 'matches'

    with open('logging.json', 'rt') as configFile:
        config = json.load(configFile)
        logging.config.dictConfig(config)

    logging.info('Start')

    try:
        os.stat(save_folder)
        try:
            list_of_files = glob.glob(save_folder + '/*.json')
            latest_file = max(list_of_files, key=os.path.getctime)

            last_match = int(
                latest_file[len(save_folder) + 1:-(len('json') + 1)])
            logging.info('Found last saved match %u', last_match)
            if first_match < last_match:
                logging.warning('Overriding first match from %u to %u',
                                first_match, last_match)
                first_match = last_match
        except Exception as e:
            logging.exception('Failed to enumerate files')
    except:
        logging.info('Creating folder \'%s\'', save_folder)
        os.mkdir(save_folder)

    current_match = first_match
    while True:
        queried_matches = query_matches(current_match, 4000)
        for match in queried_matches:
            try:
                logging.debug('Quering match %u', match)
                match_data = query_match(match)

                file_path = save_folder + '/' + str(match) + '.json'
                with open(file_path, 'w') as outfile:
                    json.dump(
                        match_data,
                        outfile,
                        sort_keys=True,
                        indent=4,
                        separators=(',', ': '))
                    logging.debug('saved match %d to %s', match, file_path)
                time.sleep(0.4)
            except Exception as e:
                logging.exception('Failed to parse match info')
        current_match = queried_matches[-1]


if __name__ == '__main__':
    main()
