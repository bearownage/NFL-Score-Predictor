from urllib.request import urlopen
from bs4 import BeautifulSoup
import re
import csv,pickle,shutil,os

def scrapeWeeklyFromYear(year):
    file = "season" + str(i) + ".pickle"
    print("season processing: " + str(i))
    page = urlopen("https://www.pro-football-reference.com/years/" + str(year) + "/games.htm")
    soup = BeautifulSoup(page, "html.parser")
    result = soup.findAll('div', attrs={'id':'all_games'})
    result = BeautifulSoup(str(result[0]), "html.parser")
    result = re.findall(r"<table(.+)</table>", str(result).replace("\n",""))
    result = "<table" + result[0] + "</table>"
    result = BeautifulSoup(result, "html.parser")
    table_body = result.find('tbody')
    rows = table_body.find_all('tr')
    head_to_head = []
    for row in rows:
        if ("class" not in row.attrs) and (row.find_all('th')[0].text.isdigit()):
            week_num = row.find_all('th')[0].text
            cols = row.find_all('td')
            home = 0
            for col in cols:
                if col.attrs["data-stat"] == "winner":
                    winner = col.text.lower().split(" ")[-1]
                elif col.attrs["data-stat"] == "loser":
                    loser = col.text.lower().split(" ")[-1]
                elif col.attrs["data-stat"] == "game_location":
                    if col.text == "@":
                        home = 1
            head_to_head.append([winner,loser,home])
    with open(file, 'wb') as handle:
        pickle.dump(head_to_head, handle, protocol=pickle.HIGHEST_PROTOCOL)

def scrapeWeeklyResults() :
    for i in range(2000,2019):
        file = "season" + str(i) + ".pickle"
        print("season processing: " + str(i))
        page = urlopen("https://www.pro-football-reference.com/years/" + str(i) + "/games.htm")
        soup = BeautifulSoup(page, "html.parser")
        result = soup.findAll('div', attrs={'id':'all_games'})
        result = BeautifulSoup(str(result[0]), "html.parser")
        result = re.findall(r"<table(.+)</table>", str(result).replace("\n",""))
        result = "<table" + result[0] + "</table>"
        result = BeautifulSoup(result, "html.parser")
        table_body = result.find('tbody')
        rows = table_body.find_all('tr')
        head_to_head = []
        for row in rows:
            if ("class" not in row.attrs) and (row.find_all('th')[0].text.isdigit()):
                week_num = row.find_all('th')[0].text
                cols = row.find_all('td')
                home = 0
                for col in cols:
                    if col.attrs["data-stat"] == "winner":
                        winner = col.text.lower().split(" ")[-1]
                    elif col.attrs["data-stat"] == "loser":
                        loser = col.text.lower().split(" ")[-1]
                    elif col.attrs["data-stat"] == "game_location":
                        if col.text == "@":
                            home = 1
                head_to_head.append([winner,loser,home])
        if os.path.exists(file):
            os.remove(file)
        with open(file, 'wb') as handle:
            pickle.dump(head_to_head, handle, protocol=pickle.HIGHEST_PROTOCOL)

def scrapeTeamsFromYear(year):
    file = "teams" + str(i) + ".pickle"
    print("teams processing: " + str(i))
    page = urlopen("https://www.pro-football-reference.com/years/" + str(year) + "/")
    soup = BeautifulSoup(page, "html.parser")
    result = soup.findAll('div', attrs={'id':'all_team_stats'})
    result = BeautifulSoup(str(result[0]), "html.parser")
    result = re.findall(r"<table(.+)</table>", str(result).replace("\n",""))
    result = "<table" + result[0] + "</table>"
    result = BeautifulSoup(result, "html.parser")
    table_body = result.find('tbody')
    rows = table_body.find_all('tr')
    teams = {}
    for row in rows:
        data = {}
        cols = row.find_all('td')
        for col in cols:
            if col.attrs["data-stat"] == "team":
                teamname = col.text.lower().split(" ")[-1]
            else:
                data[col.attrs["data-stat"]] = col.text
        teams[teamname] = data
    if os.path.exists(file):
        os.remove(file)
    with open(file, 'wb') as handle:
        pickle.dump(teams, handle, protocol=pickle.HIGHEST_PROTOCOL)

def scrapeTeamOffenses() :
    for i in range(2000,2019):
        file = "teams" + str(i) + ".pickle"
        print("teams processing: " + str(i))
        page = urlopen("https://www.pro-football-reference.com/years/" + str(i) + "/")
        soup = BeautifulSoup(page, "html.parser")
        result = soup.findAll('div', attrs={'id':'all_team_stats'})
        result = BeautifulSoup(str(result[0]), "html.parser")
        result = re.findall(r"<table(.+)</table>", str(result).replace("\n",""))
        result = "<table" + result[0] + "</table>"
        result = BeautifulSoup(result, "html.parser")
        table_body = result.find('tbody')
        rows = table_body.find_all('tr')
        teams = {}
        for row in rows:
            data = {}
            cols = row.find_all('td')
            for col in cols:
                if col.attrs["data-stat"] == "team":
                    teamname = col.text.lower().split(" ")[-1]
                else:
                    data[col.attrs["data-stat"]] = col.text
            teams[teamname] = data
        if os.path.exists(file):
            os.remove(file)
        with open(file, 'wb') as handle:
            pickle.dump(teams, handle, protocol=pickle.HIGHEST_PROTOCOL)
