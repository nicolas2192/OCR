import csv
import json
import requests


def meaning(word):
    """
    Looks up a word in the Oxford Dictionary
    :param word: String. Predicted word.
    :return: String. Word's meaning
    """
    lang_code = "en-us"
    endpoint = "entries"
    word = word.lower()

    # Retrieving credentials
    try:
        with open(".env", "r") as f:
            csv_reader = csv.reader(f)
            credentials = {line[0].split(";")[0]: line[0].split(";")[1] for line in csv_reader}
    except Exception as e:
        print("Credentials couldn't be loaded. Do you have a .env file?")
        exit()

    url = f"https://od-api.oxforddictionaries.com/api/v2/{endpoint}/{lang_code}/{word}"
    r = requests.get(url, headers={"app_id": credentials["API_ID"], "app_key": credentials["API_KEY"]})
    # print("Status code:", r.status_code)

    if r.status_code == 404:
        print(f"Your word: -- {word.upper()} -- was not found in any dictionary entry.")
        exit()
    elif r.status_code >= 400:
        print(f"Error reaching out endpoint.")
        exit()
    else:
        res = json.loads(r.text)["results"][0]["lexicalEntries"][0]["entries"][0]["senses"][0]["definitions"][0]
        res = res.capitalize()
        return res
