import bestdori
from datetime import datetime
import pandas as pd
allinfo = bestdori.events.get_all()

recent = int(list(allinfo)[-2])

_call_events_cache = {}

def identifyBand(set_input):
    if set_input == set([1,2,3,4,5]): 
        return "포피파"
    elif set_input == set([6,7,8,9,10]):
        return "앱글"
    elif set_input == set([11,12,13,14,15]):
        return "하로하피"
    elif set_input == set([16,17,18,19,20]):
        return "파스파레"
    elif set_input == set([21,22,23,24,25]):
        return "로젤리아"
    elif set_input == set([26,27,28,29,30]):
        return "모니카"
    elif set_input == set([31,32,33,34,35]):
        return "라스"
    elif set_input == set([36,37,38,39,40]):
        return '마이고'
    else:
        return '스까'

def call_events(m=-1):
    if m in _call_events_cache:
        return _call_events_cache[m]

    if m==-1: event_ids = range(1, recent+1)
    else: event_ids = range(recent-m+1, recent+1)  # 최신 이벤트부터 m번째 전까지 볼 수 있습니다.
    data = []

    for event_id in event_ids:
        event_info = allinfo.get(str(event_id))
        event_type = event_info.get("eventType", "Unknown")  # 값이 없을 경우 "Unknown"
        st = event_info.get('startAt')[0]
        et = event_info.get('endAt')[0]
        chatemp = event_info.get('characters')
        chanums = set([chatemp[j].get('characterId') for j in range(len(chatemp))])
        data.append({"id": event_id, 
                    "eventType": event_type, 
                    'eventName':event_info.get('eventName', 'Unknown')[0],
                    'startTime':(pd.to_datetime(int(st), unit = 'ms') + pd.Timedelta(hours=9)).round('5min'),
                    'endTime':(pd.to_datetime(int(et), unit = 'ms') + pd.Timedelta(hours=9)).round('5min'),
                    'Band':identifyBand(chanums),
                    })

    df = pd.DataFrame(data)
    df['eventType'] = df['eventType'].map({'medley':'메들리',
                        'festival':'팀라이브',
                        'live_try':'트라이',
                        'versus':'대반',
                        'challenge':'챌린지',
                        'mission_live':'미션',})
    _call_events_cache[m] = df
    return df

if __name__ == '__main__':
    call_event()