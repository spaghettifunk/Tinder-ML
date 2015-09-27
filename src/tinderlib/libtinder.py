#/usr/bin/env python3
import argparse
from datetime import datetime
import json
import requests
from random import randint
import sys
from time import sleep
import os
import subprocess


headers = {
    'app_version': '3',
    'platform': 'ios',
}

photopath = "photos"

fb_id = ''
fb_auth_token = ' '


class User(object):
    def __init__(self, data_dict):
        self.d = data_dict

    @property
    def user_id(self):
        return self.d['_id']

    @property
    def ago(self):
        raw = self.d.get('ping_time')
        if raw:
            d = datetime.strptime(raw, '%Y-%m-%dT%H:%M:%S.%fZ')
            secs_ago = int(datetime.now().strftime("%s")) - int(d.strftime("%s"))
            if secs_ago > 86400:
                return u'{days} days ago'.format(days=secs_ago / 86400)
            elif secs_ago < 3600:
                return u'{mins} mins ago'.format(mins=secs_ago / 60)
            else:
                return u'{hours} hours ago'.format(hours=secs_ago / 3600)

        return '[unknown]'

    @property
    def bio(self):
        try:
            x = self.d['bio'].encode('ascii', 'ignore').replace('\n', '')[:50].strip()
        except (UnicodeError, UnicodeEncodeError, UnicodeDecodeError):
            return '[garbled]'
        else:
            return x
    @property
    def name(self):
        return self.d['name']

    @property
    def age(self):
        raw = self.d.get('birth_date')
        if raw:
            d = datetime.strptime(raw, '%Y-%m-%dT%H:%M:%S.%fZ')
            return datetime.now().year - int(d.strftime('%Y'))

        return 0

    @property
    def photos(self):
        for pic in self.d['photos']:
            yield pic['url']

    def __str__(self):
        return '{name} ({age}), {distance}km, {ago}'.format(
            name=self.d['name'],
            age=self.age,
            distance=self.d['distance_mi'],
            ago=self.ago
        )

def auth_token(fb_auth_token, fb_user_id):
    h = headers
    h.update({'content-type': 'application/json'})
    req = requests.post(
        'https://api.gotinder.com/auth',
        headers=h,
        data=json.dumps({'facebook_token': fb_auth_token, 'facebook_id': fb_user_id})
    )
    try:
        return req.json()['token']
    except:
        return None

def recommendations(auth_token):
    h = headers
    h.update({'X-Auth-Token': auth_token})
    r = requests.get('https://api.gotinder.com/user/recs', headers=h)
    if r.status_code == 401 or r.status_code == 504:
        raise Exception('Invalid code')
        print(r.content)

    if 'results' not in r.json():
        print(r.json())

    for result in r.json()['results']:
        yield User(result)

def like(user_id):
    try:
        u = 'https://api.gotinder.com/like/%s' % user_id
        d = requests.get(u, headers=headers, timeout=0.7).json()
    except KeyError:
        raise
    else:
        return d['match']

def update_location(lat, lon, auth_token):
    h = headers
    h.update({'X-Auth-Token': auth_token})
    lats = "{}".format(lat)
    lons = "{}".format(lon)
    requests.post(
        'https://api.gotinder.com/user/ping',
        headers=h,
        data=json.dumps({'lat': lats, 'lon': lons})
    )

def nope(user_id):
    try:
        u = 'https://api.gotinder.com/pass/%s' % user_id
        requests.get(u, headers=headers, timeout=0.7).json()
    except KeyError:
        raise

def like_or_nope():
    return 'nope' if randint(1, 100) == 31 else 'like'

def createfolder(name):
    filepath = os.path.dirname(__file__)
    photopath = os.path.join(filepath, "photos")
    folderpath = os.path.join(photopath, name)
    if not os.path.exists(photopath):
        os.makedirs(photopath)
    if not os.path.exists(folderpath):
        os.makedirs(folderpath)
    return folderpath

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Tinder automated bot')
    parser.add_argument('-l', '--log', type=str, default='activity.log', help='Log file destination')

    args = parser.parse_args()

    print('Tinder bot')
    print('----------')
    createfolder("test")
    matches = 0
    liked = 0
    nopes = 0

    while True:
        token = auth_token(fb_auth_token, fb_id)

        if not token:
            print('could not get token')
            sys.exit(0)

        #update_location(59.326142, 17.9875454, token)  #Stockholm
        update_location(60.1687669,24.9433315, token) #Helsinki

        for user in recommendations(token):
            if not user:
                break

            print(user)
            for counter, p in enumerate(user.photos):
                print(p)
                foldername = "{}_{}".format(user.name, user.user_id)
                path = createfolder(foldername)
                subprocess.call(["wget", "-q", "-O", "{}/{}.jpg".format(path, counter), p], stdout=subprocess.DEVNULL)

            # try:
            #     action = like_or_nope()
            #     if action == 'like':
            #         print(' -> Like')
            #         match = like(user.user_id)
            #         if match:
            #             print(' -> Match!')
            #
            #         with open('./liked.txt', 'a') as f:
            #             f.write(user.user_id + u'\n')
            #
            #     else:
            #         print(' -> random nope :(')
            #         nope(user.user_id)
            #
            # except:
            #     print('networking error %s' % user.user_id)

            s = float(randint(250, 2500) / 1000)
            sleep(s)
