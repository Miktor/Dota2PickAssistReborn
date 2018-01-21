import json
import datetime
import os
import argparse
import glob
import time
import logging
import logging.config
import urllib.request
import urllib.parse


class OpenDotaRequester(object):
    API_PATH = 'https://api.opendota.com/api/'
    MAX_TRIES = 25
    DELAY_AFTER_FAILED_RESPONSE = 5.0
    STATS_INTERVAL_MINUTES = 1

    def __init__(self, interval):
        self.interval = interval
        self.timestamp = datetime.datetime.min
        self.timestamp_stat = None
        self.query_count = 0
        self.stat_interval = datetime.timedelta(minutes=self.STATS_INTERVAL_MINUTES)
        self.logger = logging.getLogger('Requester')

    def query(self, query_text):
        now = datetime.datetime.utcnow()
        if now < self.timestamp + self.interval:
            time.sleep(self.timestamp + self.interval - now)

        self.timestamp = now

        tries = 0
        response = None
        while tries < self.MAX_TRIES:
            try:
                tries += 1
                response = urllib.request.urlopen(query_text)
                if response.code is not 200:
                    self.logger.warning('Failed to query, response code = %d', response.code)
                    time.sleep(self.DELAY_AFTER_FAILED_RESPONSE)
                    self.logger.warning('Retrying: try %d out of %d', tries, self.MAX_TRIES)
                else:
                    break
            except Exception as e:
                self.logger.error('Failed to query = %s, error = %s', query_text, e.msg)
                time.sleep(5)

        if response is None:
            self.logger.warning('Failed to get a valid response after %d tries', tries)
            return None

        data = json.loads(response.read().decode("utf-8"))

        self.query_count += 1

        if self.timestamp_stat is None:
            self.timestamp_stat = now
        elif now > self.timestamp_stat + self.stat_interval:
            self.logger.info('Query rate = %d/%s', self.query_count, str(self.stat_interval))
            self.timestamp_stat = now
            self.query_count = 0
        return data

    def query_matches(self, starting_match_id, min_mmr):
        try:
            sql_query = ("select * from public_matches "
                         "where avg_mmr >= {0} and match_id > {1} "
                         "and lobby_type = 7 and game_mode in (2, 22) "
                         "ORDER BY start_time asc "
                         "limit 20")

            query_text = self.API_PATH + 'explorer?sql=' + urllib.parse.quote(
                sql_query.format(min_mmr, starting_match_id))

            self.logger.debug(
                'Quering match from mmr %d, min match id = %d, string = %s', min_mmr,
                starting_match_id, query_text)

            data = self.query(query_text)

            rows = data['rows']
            matches = []
            for row in rows:
                match_id = row['match_id']
                mmr = row['avg_mmr']
                duration = row['duration']

                self.logger.debug(
                    'match_id = %d, mmr = %d, date = %s, duration = %d', match_id, mmr,
                    str(datetime.datetime.fromtimestamp(row['start_time'])), duration / 60)

                match = {'match_id': match_id, 'user_info': {'mmr': mmr}}
                matches.append(match)

            return matches
        except Exception:
            self.logger.exception('Failed to query match info')
            return []

    def query_match(self, match_id):
        query = self.API_PATH + "matches/{}".format(match_id)
        self.logger.debug('Querying match %d, string = %s', match_id, query)
        return self.query(query)


def main(first_match: int, save_folder: str, min_mmr: int, requester_delay: int):
    logging.info('Start')

    if os.path.exists(save_folder) and os.path.isdir(save_folder):
        try:
            list_of_files = glob.glob(save_folder + '/*.json')
            latest_file = max(list_of_files, key=os.path.getctime)

            last_match = int(os.path.splitext(os.path.basename(latest_file))[0])

            logging.info('Found last saved match %u', last_match)
            if first_match < last_match:
                logging.warning('Overriding first match from %u to %u', first_match, last_match)
                first_match = last_match
        except Exception:
            logging.exception('Failed to enumerate files')
    else:
        logging.info('Creating folder \'%s\'', save_folder)
        os.mkdir(save_folder)

    requester = OpenDotaRequester(datetime.timedelta(milliseconds=requester_delay))

    current_match = first_match
    while True:
        queried_matches = requester.query_matches(current_match, min_mmr)
        for match_info in queried_matches:
            try:
                match_id = match_info['match_id']
                logging.debug('Quering match %u', match_id)
                match_data = requester.query_match(match_id)
                match_data.update(match_info['user_info'])

                file_path = save_folder + '/' + str(match_id) + '.json'
                with open(file_path, 'w') as outfile:
                    json.dump(
                        match_data,
                        outfile,
                        indent=4,
                        separators=(',', ': '))
                    logging.debug('saved match %d to %s', match_id, file_path)
            except Exception as e:
                logging.exception('Failed to parse match info')
        current_match = queried_matches[-1]['match_id']


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('output_dir', type=str, help='Output matches directory')
    parser.add_argument('--first_match', type=int, default=3592625302, help='Starting match ID')
    parser.add_argument('--min_mmr', type=int, default=4000, help='Minimal mmr')
    parser.add_argument('--delay', type=int, default=350, help='Delay between requests')
    parser.add_argument('--logging_config', type=str, default='logging.json', help='Path to logging configuration')
    args = parser.parse_args()

    if os.path.exists(args.logging_config):
        with open(args.logging_config, 'rt') as configFile:
            config = json.load(configFile)
            logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=logging.INFO)
        logging.debug('Failed to locate the logging config \'%s\' using basic settings', args.logging_config)

    main(args.first_match, args.output_dir, args.min_mmr, args.delay)
