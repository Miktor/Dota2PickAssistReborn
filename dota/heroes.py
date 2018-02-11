import enum


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


NUM_HEROES = len(Hero)
assert NUM_HEROES == 115


def get_dota_hero_id(hero_id):
    for idx, e in enumerate(list(Hero)):
        if e.value == hero_id:
            return idx
    raise RuntimeError('No such hero {0}'.format(hero_id))
