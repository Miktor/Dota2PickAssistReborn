import json
import enum

HEROES = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 25, 26, 27, 28, 29, 30, 31, 32,
    33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61,
    62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90,
    91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 119,
    120
]

NUM_HEROES = len(HEROES)
assert NUM_HEROES == 115


class Hero(enum.IntEnum):
    AntiMage = 1
    Axe = 2
    Bane = 3
    Bloodseeker = 4
    CrystalMaiden = 5
    DrowRanger = 6
    Earthshaker = 7
    Juggernaut = 8
    Mirana = 9
    Morphling = 10
    ShadowFiend = 11
    PhantomLancer = 12
    Puck = 13
    Pudge = 14
    Razor = 15
    SandKing = 16
    StormSpirit = 17
    Sven = 18
    Tiny = 19
    VengefulSpirit = 20
    Windranger = 21
    Zeus = 22
    Kunkka = 23
    Lina = 25
    Lion = 26
    ShadowShaman = 27
    Slardar = 28
    Tidehunter = 29
    WitchDoctor = 30
    Lich = 31
    Riki = 32
    Enigma = 33
    Tinker = 34
    Sniper = 35
    Necrophos = 36
    Warlock = 37
    Beastmaster = 38
    QueenofPain = 39
    Venomancer = 40
    FacelessVoid = 41
    WraithKing = 42
    DeathProphet = 43
    PhantomAssassin = 44
    Pugna = 45
    TemplarAssassin = 46
    Viper = 47
    Luna = 48
    DragonKnight = 49
    Dazzle = 50
    Clockwerk = 51
    Leshrac = 52
    NaturesProphet = 53
    Lifestealer = 54
    DarkSeer = 55
    Clinkz = 56
    Omniknight = 57
    Enchantress = 58
    Huskar = 59
    NightStalker = 60
    Broodmother = 61
    BountyHunter = 62
    Weaver = 63
    Jakiro = 64
    Batrider = 65
    Chen = 66
    Spectre = 67
    AncientApparition = 68
    Doom = 69
    Ursa = 70
    SpiritBreaker = 71
    Gyrocopter = 72
    Alchemist = 73
    Invoker = 74
    Silencer = 75
    OutworldDevourer = 76
    Lycan = 77
    Brewmaster = 78
    ShadowDemon = 79
    LoneDruid = 80
    ChaosKnight = 81
    Meepo = 82
    TreantProtector = 83
    OgreMagi = 84
    Undying = 85
    Rubick = 86
    Disruptor = 87
    NyxAssassin = 88
    NagaSiren = 89
    KeeperoftheLight = 90
    Io = 91
    Visage = 92
    Slark = 93
    Medusa = 94
    TrollWarlord = 95
    CentaurWarrunner = 96
    Magnus = 97
    Timbersaw = 98
    Bristleback = 99
    Tusk = 100
    SkywrathMage = 101
    Abaddon = 102
    ElderTitan = 103
    LegionCommander = 104
    Techies = 105
    EmberSpirit = 106
    EarthSpirit = 107
    Underlord = 108
    Terrorblade = 109
    Phoenix = 110
    Oracle = 111
    WinterWyvern = 112
    ArcWarden = 113
    MonkeyKing = 114
    DarkWillow = 119
    Pangolier = 120


class Lane(enum.Enum):
    Bot = 1
    Mid = 2
    Top = 3
    RadiantForest = 4
    DireForest = 5


class Role(enum.Enum):
    Carry = 1
    Support = 2
    Offlane = 3
    RadiantForest = 4
    DireForest = 5


def encode_hero(hero_id):
    idx = HEROES.index(hero_id)
    if idx >= 0:
        return idx
    raise RuntimeError('No such hero {0}'.format(hero_id))


def load_heroes():
    with open('model/heroes.json', 'r') as f:
        return json.loads(f.read())


#heroes_data = load_heroes()

if __name__ == '__main__':
    pass
