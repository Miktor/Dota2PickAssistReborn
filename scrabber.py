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


def query_matches(startingMatchId, minMMR):
    try:
        sqlQuery = ("select * from public_matches "
                    "where avg_mmr >= {0} and match_id > {1} "
                    "and lobby_type = 7 and game_mode in (2, 22) "
                    "ORDER BY start_time asc "
                    "limit 100")

        query = "https://api.opendota.com/api/explorer?sql=" + urllib.parse.quote(
            sqlQuery.format(minMMR, startingMatchId))

        logging.debug(
            'Quering match from mmr %d, min match id = %d, string = %s', minMMR,
            startingMatchId, query)

        response = urllib.request.urlopen(query)

        data = json.loads(response.read())

        rows = data['rows']
        matches = []
        for row in rows:
            matchID = row['match_id']
            matches.append(matchID)
            logging.debug(
                'match_id = %d, mmr = %d, date = %s', matchID, row['avg_mmr'],
                str(datetime.datetime.fromtimestamp(row['start_time'])))
        return matches
    except Exception as e:
        logging.exception('Failed to enumerate files')
        return []


def query_match(matchId):
    try:
        query = "https://api.opendota.com/api/matches/{}".format(matchId)
        logging.debug('Quering match %d, string = %s', matchId, query)

        response = urllib.request.urlopen(query)

        data = json.loads(response.read())
        return data
    except Exception as e:
        logging.exception('Failed to enumerate files')


def main():
    firstMatch = 3534655441
    saveFolder = 'matches'

    with open('logging.json', 'rt') as configFile:
        config = json.load(configFile)
        logging.config.dictConfig(config)

    logging.info('Start')

    try:
        os.stat(saveFolder)
        try:
            list_of_files = glob.glob(saveFolder + '/*.json')
            latest_file = max(list_of_files, key=os.path.getctime)

            lastMatch = int(
                latest_file[len(saveFolder) + 1:-(len('json') + 1)])
            logging.info('Found last saved match %u', lastMatch)
            if firstMatch < lastMatch:
                logging.warning('Overriding first match from %u to %u',
                                firstMatch, lastMatch)
                firstMatch = lastMatch
        except Exception as e:
            logging.exception('Failed to enumerate files')
    except:
        logging.info('Creating folder \'%s\'', saveFolder)
        os.mkdir(saveFolder)

    while True:
        for match in query_matches(firstMatch, 4000):
            try:
                logging.debug('Quering match %u', match)
                matchData = query_match(match)

                filePath = saveFolder + '/' + str(match) + '.json'
                with open(filePath, 'w') as outfile:
                    json.dump(
                        matchData,
                        outfile,
                        sort_keys=True,
                        indent=4,
                        separators=(',', ': '))
                    logging.debug('saved match %d to %s', match, filePath)
                time.sleep(0.25)
            except Exception as e:
                logging.exception('Failed to enumerate files')


if __name__ == '__main__':
    main()
