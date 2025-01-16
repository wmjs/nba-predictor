import pandas as pd
import re
import numpy as np

def get_odds(odds_type, abv):
    # df = pd.read_html(f'https://www.sportsline.com/nba/odds/{odds_type}')[0].drop(columns=["Proj Score", "bet365newjersey"]).rename(columns={"Unnamed: 3": "mgm"})[["Matchup", "draftkings"]]
    df = pd.read_html(f'https://www.sportsline.com/nba/odds/{odds_type}')[0]
    print(" ")
    print(odds_type)
    print(df.columns)
    print(" ")
    try:
        df = df[["Matchup", "consensus"]]
    except:
        print("FAILED TRYING AGAIN")
        df = pd.read_html(f'https://www.sportsline.com/nba/odds/{odds_type}')[0]
        df = df[["Matchup", "consensus"]]
        

    df.dropna(inplace=True, how='any', axis=0)
    df = df[~df["Matchup"].str.contains("Advanced")].reset_index(drop=True)

    df_grouped = df.groupby(df.index // 3).agg(lambda x: list(x)).reset_index(drop=True)
    def map_to_dict(lst):
        return {
            "Away": lst[0],
            "Home": lst[1], 
            "Time": lst[2]
        }

    df_grouped = df_grouped.apply(lambda x: x.map(map_to_dict))

    for col in df_grouped.columns:
        if not isinstance(df_grouped[col].iloc[0], dict):
            continue
            
        for key in df_grouped[col].iloc[0].keys():
            df_grouped[f"{key}-{col}"] = df_grouped[col].apply(lambda x: x[key])
        
        df_grouped.drop(columns=[col], inplace=True)

    df_grouped["Away-Matchup"] = df_grouped["Away-Matchup"].apply(lambda x: "".join([c for c in x if c.isalpha()]))
    df_grouped["Home-Matchup"] = df_grouped["Home-Matchup"].apply(lambda x: "".join([c for c in x if c.isalpha()]))
    df_grouped.loc[df_grouped["Away-Matchup"] == "ers", "Away-Matchup"] = "76ers"
    df_grouped.loc[df_grouped["Home-Matchup"] == "ers", "Home-Matchup"] = "76ers"

    def clean_matchup_date(entry):
        entry = entry[:entry.find("UTC") + 3]
        entry = re.sub(r'\d+\s*Expert\s*Picks?', '', entry)
        current_year = pd.Timestamp.now().year
        current_month = pd.Timestamp.now().month
        if current_month < 10:
            if entry.split(" ")[0] in ["Oct", "Nov", "Dec"]:
                year = current_year - 1
            else:
                year = current_year
        else:
            if entry.split(" ")[0] in ["Oct", "Nov", "Dec"]:
                year = current_year
            else:
                year = current_year + 1
        entry = f"{entry} {year}"
        return pd.to_datetime(entry).tz_convert('US/Eastern').date()

    df_grouped["Date"] = df_grouped["Time-Matchup"].apply(clean_matchup_date)
    df_grouped.rename(columns={"Away-Matchup": "Away", "Home-Matchup": "Home", "Away-consensus": f"Away odds", "Home-consensus": f"Home odds"}, inplace=True)
    df_grouped = df_grouped.drop(columns=["Time-Matchup", "Time-consensus"]).reset_index(drop=True)

    def clean_odds(row, hoa, odds_type):
        entry = row[f"{hoa} odds"]
        entry = entry[:entry.find("Open")]

        pos = re.search(r'[+-][^+-]*$', entry).start()
        line = entry[:pos]
        try:
            odd = float(entry[pos:])
        except:
            return None, None, None

        if odd < 0:
            implied_prob = round((np.abs(odd) / (np.abs(odd) + 100)) * 100, 2)
        else:
            implied_prob = round(((100 / (np.abs(odd) + 100)) * 100), 2)
        
        if odds_type == "money-line":
            return row[hoa], odd, implied_prob
        else:
            return line, odd, implied_prob
        
    nice_odds = {"money-line": "Money Line", "picks-against-the-spread": "Spread", "over-under": "Over/Under"}
    
    df_grouped["Away Line"] = df_grouped.apply(lambda row: clean_odds(row, hoa="Away", odds_type=odds_type), axis=1)
    df_grouped["Home Line"] = df_grouped.apply(lambda row: clean_odds(row, hoa="Home", odds_type=odds_type), axis=1)

    df_grouped.drop(columns=["Away odds", "Home odds"], inplace=True)

    df_grouped[[f"Away {nice_odds[odds_type]}", f"Away {nice_odds[odds_type]} Odds", f"Away {nice_odds[odds_type]} Implied Probability"]] = pd.DataFrame(df_grouped["Away Line"].tolist(), index=df_grouped.index)
    df_grouped[[f"Home {nice_odds[odds_type]}", f"Home {nice_odds[odds_type]} Odds", f"Home {nice_odds[odds_type]} Implied Probability"]] = pd.DataFrame(df_grouped["Home Line"].tolist(), index=df_grouped.index)

    df_grouped = df_grouped[["Away", "Home", "Date", 
                            f"Away {nice_odds[odds_type]}", f"Away {nice_odds[odds_type]} Odds", f"Away {nice_odds[odds_type]} Implied Probability",
                            f"Home {nice_odds[odds_type]}", f"Home {nice_odds[odds_type]} Odds", f"Home {nice_odds[odds_type]} Implied Probability"]]
    
    df_grouped.loc[df_grouped["Away"] == "TrailBlazers", "Away"] = "Trail Blazers"
    df_grouped.loc[df_grouped["Home"] == "TrailBlazers", "Home"] = "Trail Blazers"

    df_grouped["Date"] = pd.to_datetime(df_grouped["Date"])

    out = df_grouped.merge(abv, left_on = "Away", right_on = "Team Name", how = "left").drop(columns = ["Abbreviation", "Team Name", "Away"]).rename(columns = {"Team": "Away"})\
    .merge(abv, left_on = "Home", right_on = "Team Name", how = "left").drop(columns = ["Abbreviation", "Team Name", "Home"]).rename(columns = {"Team": "Home"})        


    return out


def load_odds(odds_type = ""):

    if odds_type == "":
        types = ["money-line", "picks-against-the-spread", "over-under"]
    else:
        types = odds_type.split(",")


    out = []
    abv = pd.read_csv("nba_data/nba_team_abbreviations.csv")
    abv["Team Name"] = abv["Team"].apply(lambda x: x.split(" ")[-1])
    abv.loc[abv["Team Name"] == "Blazers", "Team Name"] = "Trail Blazers"

    for odds_type in types:
        out.append(get_odds(odds_type, abv))

    return out