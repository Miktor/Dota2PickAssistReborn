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


class OpenDotaRequester(object):
    interval = None
    timestamp = None
    query_count = 0
    timestamp_stat = None
    stat_interval = datetime.timedelta(minutes=1)

    def __init__(self, interval):
        self.interval = interval
        self.timestamp = datetime.datetime.min
        self.timestamp_stat = None

    def query(self, query_text):
        now = datetime.datetime.utcnow()
        if now < self.timestamp + self.interval:
            time.sleep(self.timestamp + self.interval - now)

        self.timestamp = now

        tries = 0
        while tries < 25:
            try:
                tries += 1
                response = urllib.request.urlopen(query_text)
                if response.code is not 200:
                    logging.warning(
                        'Failed to query, return code = %d', response.code)
                    time.sleep(5)
                    logging.warning('Retrying')
                else:
                    break
            except Exception as e:
                logging.error('Failed to query = %s, error = %s',
                              query_text, e.msg)
                time.sleep(5)

        data = json.loads(response.read().decode("utf-8"))

        self.query_count += 1

        if self.timestamp_stat is None:
            self.timestamp_stat = now
        elif now > self.timestamp_stat + self.stat_interval:
            logging.info('query rate = %d per %s',
                         self.query_count, str(self.stat_interval))
            self.timestamp_stat = now
            self.query_count = 0
        return data


open_dota_requester = OpenDotaRequester(datetime.timedelta(milliseconds=350))


def query_matches(starting_match_id, min_mmr):
    try:
        sql_query = ("select * from public_matches "
                     "where avg_mmr >= {0} and match_id > {1} "
                     "and lobby_type = 7 and game_mode in (2, 22) "
                     "ORDER BY start_time asc "
                     "limit 20")

        query_text = "https://api.opendota.com/api/explorer?sql=" + urllib.parse.quote(
            sql_query.format(min_mmr, starting_match_id))

        logging.debug(
            'Quering match from mmr %d, min match id = %d, string = %s', min_mmr,
            starting_match_id, query_text)

        data = open_dota_requester.query(query_text)

        rows = data['rows']
        matches = []
        for row in rows:
            match_id = row['match_id']
            mmr = row['avg_mmr']
            duration = row['duration']

            logging.debug(
                'match_id = %d, mmr = %d, date = %s, duration = %d', match_id, mmr,
                str(datetime.datetime.fromtimestamp(row['start_time'])), duration / 60)

            match = {'match_id': match_id, 'user_info': {'mmr': mmr}}

            matches.append(match)

        return matches
    except Exception:
        logging.exception('Failed to query match info')
        return []


def query_match(match_id):

    query_text = "https://api.opendota.com/api/matches/{}".format(match_id)
    logging.debug('Quering match %d, string = %s', match_id, query_text)

    return open_dota_requester.query(query_text)


def main():
    first_match = 3592625302
    save_folder = 'raw'

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
        except Exception:
            logging.exception('Failed to enumerate files')
    except:
        logging.info('Creating folder \'%s\'', save_folder)
        os.mkdir(save_folder)

    current_match = first_match
    while True:
        queried_matches = query_matches(current_match, 6000)
        for match_info in queried_matches:
            try:
                match_id = match_info['match_id']
                logging.debug('Quering match %u', match_id)
                match_data = query_match(match_id)
                match_data.update(match_info['user_info'])

                file_path = save_folder + '/' + str(match_id) + '.json'
                with open(file_path, 'w') as outfile:
                    json.dump(
                        match_data,
                        outfile,
                        indent=4,
                        separators=(',', ': '))
                    logging.debug('saved match %d to %s',
                                  match_id, file_path)

            except Exception as e:
                logging.exception('Failed to parse match info')
        current_match = queried_matches[-1]['match_id']


if __name__ == '__main__':
    main()
