from typing import List, Union
import enum

from dota.heroes import Hero


class Side(enum.IntEnum):
    DIRE = 0
    RADIANT = 1


class Lane(enum.IntEnum):
    Bot = 0
    Mid = 1
    Top = 2
    RadiantForest = 3
    DireForest = 4
    Count = 5


class Role(enum.IntEnum):
    Carry = 0
    Support = 1
    Offlane = 2
    RadiantForest = 3
    DireForest = 4
    Count = 5


class PickPhase(enum.IntEnum):
    RADIANT_PICK = 0
    DIRE_PICK = 1
    RADIANT_BAN = 2
    DIRE_BAN = 3


class PickedHero(object):
    def __init__(self, hero: Hero, lane: Lane, role: Role):
        self.hero = hero
        self.lane = lane
        self.role = role

    def __str__(self):
        return '{0} ({1}, {2})'.format(self.hero.name, self.lane.name, self.role.name)


class Pick(object):
    def __init__(self,
                 radiant_heroes: List[Union[PickedHero, Hero]] = None,
                 dire_heroes: List[Union[PickedHero, Hero]] = None):
        self.radiant = []  # type: List[PickedHero]
        self.dire = []  # type: List[PickedHero]
        self.heroes = []  # type: List[Hero]

        if radiant_heroes is not None:
            for hero in radiant_heroes:
                self.append(hero, Side.RADIANT)

        if dire_heroes is not None:
            for hero in dire_heroes:
                self.append(hero, Side.DIRE)

    def append(self, hero: Union[Hero, PickedHero], side: Side):
        collection = self.radiant if side is Side.RADIANT else self.dire
        if len(collection) == 5:
            raise RuntimeError('Pick for side {0} already contains 5 heroes'.format(side.name))

        if isinstance(hero, Hero):
            picked_hero = PickedHero(hero)
        elif isinstance(hero, PickedHero):
            picked_hero = hero
        else:
            raise RuntimeError('Hero or PickedHero expected')

        if picked_hero.hero in self.heroes:
            raise RuntimeError('Hero {0} is already in pick'.format(picked_hero))

        self.heroes.append(picked_hero.hero)
        collection.append(picked_hero)

    def is_complete(self) -> bool:
        return len(self.radiant) == 5 and len(self.dire) == 5

    def has_hero(self, hero: Hero) -> bool:
        return hero in self.heroes

    def __str__(self):
        radiant = ', '.join(str(h) for h in self.radiant)
        dire = ', '.join(str(h) for h in self.dire)
        return 'Pick:\n\tRadiant:\t{0}\n\tDire:\t{1}'.format(radiant, dire)


class Match(object):
    DEFAULT_DURATION = 60 * 45.0  # 45 min as 1.0
    DEFAULT_MMR = None

    def __init__(self,
                 pick: Pick,
                 winning_side: Side,
                 duration: float=DEFAULT_DURATION,
                 mmr: float=DEFAULT_MMR):
        self.pick = pick
        self.winning_side = winning_side
        self.duration = duration
        self.mmr = mmr


def flip_side(phase: PickPhase):
    if phase is PickPhase.DIRE_BAN:
        return phase.RADIANT_BAN
    if phase is PickPhase.RADIANT_BAN:
        return phase.DIRE_BAN
    if phase is PickPhase.RADIANT_PICK:
        return phase.DIRE_PICK
    if phase is PickPhase.DIRE_PICK:
        return phase.RADIANT_PICK
    raise RuntimeError('Unexpected phase: {0}'.format(phase))


CM_SCHEDULE_RADIANT_FIRST = [
    # Ban phase
    PickPhase.RADIANT_BAN,
    PickPhase.DIRE_BAN,
    PickPhase.RADIANT_BAN,
    PickPhase.DIRE_BAN,
    PickPhase.RADIANT_BAN,
    PickPhase.DIRE_BAN,

    # First picks
    PickPhase.RADIANT_PICK,
    PickPhase.DIRE_PICK,
    PickPhase.DIRE_PICK,
    PickPhase.RADIANT_PICK,

    # Second bans
    PickPhase.DIRE_BAN,
    PickPhase.RADIANT_BAN,
    PickPhase.DIRE_BAN,
    PickPhase.RADIANT_BAN,

    # Second picks
    PickPhase.DIRE_PICK,
    PickPhase.RADIANT_PICK,
    PickPhase.DIRE_PICK,
    PickPhase.RADIANT_PICK,

    # Last ban
    PickPhase.DIRE_BAN,
    PickPhase.RADIANT_BAN,

    # Last pick
    PickPhase.RADIANT_PICK,
    PickPhase.DIRE_PICK,
]

CM_SCHEDULE_DIRE_FIRST = [flip_side(phase) for phase in CM_SCHEDULE_RADIANT_FIRST]


class Action(object):
    pass


class PickAction(Action):
    def __init__(self, picked_hero: PickedHero):
        self.picked_hero = picked_hero

    def __str__(self):
        return 'Action[Pick]: {0}'.format(self.picked_hero)


class BanAction(Action):
    def __init__(self, hero: Hero):
        self.hero = hero

    def __str__(self):
        return 'Action[Ban]: {0}'.format(self.hero.name)


## ACTIONS_PICK = [PickAction(PickedHero(hero)) for hero in Hero]
## ACTIONS_BAN = [BanAction(hero) for hero in Hero]
## ACTION_SPACE = ACTIONS_PICK + ACTIONS_BAN


class PickGame(object):
    def do_action(self, action: Action):
        raise NotImplementedError

    def get_legal_actions(self) -> List[Action]:
        raise NotImplementedError

    def is_completed(self) -> bool:
        raise NotImplementedError


class CaptainsModePickGame(PickGame):
    def __init__(self, start_side: Side = Side.RADIANT):
        if start_side is Side.RADIANT:
            self.schedule = CM_SCHEDULE_RADIANT_FIRST
        else:
            self.schedule = CM_SCHEDULE_DIRE_FIRST

        self.pick = Pick()
        self.banned = []
        self.step = 0

    def do_action(self, action: Action):
        if self.is_completed():
            raise RuntimeError('Cannot do action: pick game is completed')

        phase = self.schedule[self.step]
        if isinstance(action, PickAction):
            if action.picked_hero.hero in self.banned:
                raise RuntimeError('Cannot pick hero that is already banned: {0}'.format(action.picked_hero.hero))
            if phase is PickPhase.RADIANT_PICK:
                self.pick.append(action.picked_hero, Side.RADIANT)
            elif phase is PickPhase.DIRE_PICK:
                self.pick.append(action.picked_hero, Side.DIRE)
            else:
                raise RuntimeError('Incorrect phase: {0}'.format(phase))
        elif isinstance(action, BanAction):
            hero = action.hero
            if self.pick.has_hero(hero):
                raise RuntimeError('Cannot ban hero that is already picked: {0}'.format(hero))
            if hero in self.banned:
                raise RuntimeError('Hero is already banned: {0}'.format(hero.name))
            self.banned.append(action.hero)
        else:
            raise RuntimeError('Incorrect action: {0}'.format(action))
        self.step += 1

    def get_legal_actions(self) -> List[Action]:
        if self.is_completed():
            return []

        phase = self.schedule[self.step]
        actions = []  # type: List[Action]
        if phase is PickPhase.RADIANT_PICK or phase is PickPhase.DIRE_PICK:
            for a in ACTIONS_PICK:
                hero = a.picked_hero.hero
                if not self.pick.has_hero(hero) and hero not in self.banned:
                    actions.append(a)
            return actions
        elif phase is PickPhase.RADIANT_BAN or phase is PickPhase.DIRE_BAN:
            for a in ACTIONS_BAN:
                hero = a.hero
                if not self.pick.has_hero(hero) and hero not in self.banned:
                    actions.append(a)
            return actions
        else:
            raise RuntimeError('Incorrect phase: {0}'.format(phase))

    def is_completed(self):
        return self.step >= len(self.schedule)
