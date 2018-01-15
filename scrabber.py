import json, datetime, os, sys, glob
import time
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
        print(query)

        response = urllib.request.urlopen(query)

        data = json.loads(response.read())
        print(
            json.dumps(data, sort_keys=True, indent=4, separators=(',', ': ')))
        rows = data['rows']
        rows = data['rows']
        matches = []
        for row in rows:
            matchID = row['match_id']
            matches.append(matchID)
            print('match_id = {}, mmr = {}, date = {}'.format(
                matchID, row['avg_mmr'],
                datetime.datetime.fromtimestamp(row['start_time'])))
        return matches
    except:
        print("Unexpected error:", sys.exc_info()[0])
        return []


def query_match(matchId):
    try:
        query = "https://api.opendota.com/api/matches/{}".format(matchId)
        print(query)

        response = urllib.request.urlopen(query)

        data = json.loads(response.read())
        return data
    except:
        print("Unexpected error:", sys.exc_info()[0])


firstMatch = 3534655441
saveFolder = 'matches'

try:
    os.stat(saveFolder)
    try:
        list_of_files = glob.glob(saveFolder + '/*.json')
        latest_file = max(list_of_files, key=os.path.getctime)

        lastMatch = int(latest_file[len(saveFolder) + 1:-(len('json') + 1)])
        print('Found last saved match {}'.format(lastMatch))
        if firstMatch < lastMatch:
            print('Overriding first match from {} to {}'.format(
                firstMatch, lastMatch))
            firstMatch = lastMatch
    except:
        print("Unexpected error:", sys.exc_info()[0])
except:
    os.mkdir(saveFolder)

while True:
    for match in query_matches(firstMatch, 4000):
        try:
            print("Quering match {}".format(match))
            matchData = query_match(match)

            filePath = saveFolder + '/' + str(match) + '.json'
            with open(filePath, 'w') as outfile:
                json.dump(
                    matchData,
                    outfile,
                    sort_keys=True,
                    indent=4,
                    separators=(',', ': '))
                print('saved to {}'.format(filePath))
            time.sleep(0.25)
        except:
            print("Unexpected error:", sys.exc_info()[0])
