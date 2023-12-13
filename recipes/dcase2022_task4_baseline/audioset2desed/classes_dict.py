"""
AudioSet (strong)의 클래스 중 DESED 이벤트에 대응하는 클래스들을 맵핑하고, 타겟 레이블 학습에 도움이 될 추가 레이블을 정의함
AudioSet2DESED 연구노트
https://www.notion.so/rtzr/AudioSet2DESED-cf311232d1d3420a9cfd026b6a79b8c7
"""

from collections import OrderedDict


mapping = OrderedDict(
    {
        "Alarm_bell_ringing": [
            "Fire alarm",
            "Alarm",
            "Alarm clock",
            "Smoke detector, smoke alarm",
            "Car alarm",
            "Siren",
            "Ambulance (siren)",
            "Fire engine, fire truck (siren)",
            "Police car (siren)",
            "Change ringing (campanology)",
            "Carbon monoxide detector, CO detector",
            "Ding-dong",
            "Ding",
            "Chime",
            "Wind chime",
            "Jingle bell",
            "Buzzer",
            "Bell",
            "Church bell",
            "Doorbell",
            "Mechanical bell",
            "Telephone bell ringing",
            "Bicycle bell",
        ],
        "Blender": ["Blender, food processor"],
        "Cat": ["Cat"],
        "Dishes": ["Dishes, pots, and pans", "Cutlery, silverware", "Knife", "Chopping (food)", "Glass chink, clink"],
        "Dog": ["Dog", "Canidae, dogs, wolves", "Whimper (dog)"],
        "Electric_shaver_toothbrush": [
            "Electric toothbrush",
            "Electric shaver, electric razor",
            "Dental drill, dentist's drill",
        ],
        "Frying": ["Frying (food)", "Sizzle"],
        "Running_water": [
            "Water",
            "Water tap, faucet",
            "Waterfall",
            "Sink (filling or washing)",
            "Liquid",
            "Pump (liquid)",
            "Fill (with liquid)",
            "Toilet flush",
            "Bathtub (filling or washing)",
            "Pour",
        ],
        "Speech": [
            "Speech",
            "Child speech, kid speaking",
            "Female speech, woman speaking",
            "Male speech, man speaking",
        ],
        "Vacuum_cleaner": ["Vacuum cleaner"],
        # additional classes
        # "Music": ["Music"],
        # "Traffic_noise": ["Traffic noise, roadway noise"],
        # "Rural_natural_noise": ["Outside, rural or natural"],
        # "Bird": ["Bird", "Bird vocalization, bird call, bird song"],
        # "Silence": ["Silence"],
    }
)
