# Make floating-point division the default
from __future__ import division

from operator import iand, mul
from wumpus_world import Contents, Senses, Actions, Directions


## Agent ##
class Agent:
  def action(self, known_world, rng):
    raise NotImplementedError()

## HumanAgent ##
class HumanAgent(Agent):
  # Action and Direction abbreviations
  act_abbrev = dict((n, n.replace("MOVE", "M").replace("SHOOT", "S")) for n, v in Actions)
  dir_abbrev = dict((n, n.replace("NORTH", "N").replace("SOUTH", "S").replace("EAST", "E").replace("WEST", "W")) for n, v in Directions)

  def action(self, known_world, rng):
    # Ask a human what to do. What could go wrong?
    action = self.__parse_input(raw_input("What do you want to do?\n"))
    while action is None:
      print "That isn't a valid action"
      action = self.__parse_input(raw_input("What do you want to do?\n"))
    return action

  def __parse_input(self, input):
    # Make sure the split array is the right size
    input = input.split()
    if len(input) != 2:
      return None

    # Look for the full or abbreviated action
    action = None
    act = input[0].strip().upper()
    for full, abbrev in self.act_abbrev.iteritems():
      if act == full or act == abbrev:
        action = getattr(Actions, full)
    if action is None:
      return None

    # Look for the full or abbreviated direction
    direction = None
    dir = input[1].strip().upper()
    for full, abbrev in self.dir_abbrev.iteritems():
      if dir == full or dir == abbrev:
        direction = getattr(Directions, full)
    if direction is None:
      return None

    return action, direction


def cached_prob(dist_func):
  def _cached_prob(self, room, known_world):
    # Retrieve or recompute the distribution
    move_num = known_world.move_num()
    cache = known_world.get(dist_func, (-1, None))
    if cache[0] == move_num:
      dist = cache[1]
    else:
      dist = dist_func(self, known_world)
      known_world[dist_func] = (move_num, dist)

    # Look up the probability
    if room not in dist:
      raise ValueError("Unexplored room %d" % room)
    return dist[room]
  return _cached_prob


DEBUGFLAGS = [
  'DEBUG DANGER: ',
  ]
DEBUG = 'DEBUG DANGER: '
DEBUG = 1
def pprint(*args):
  if not DEBUG or DEBUG:
    return
  string = ''
  for thing in args:
    string += str(thing)
  if args[0] in DEBUGFLAGS:
    print string
  elif DEBUG is args[0]:
    print string

#### Global constants ####
# Senses (evidence variables)
STENCH = 'STENCH'
BREEZE = 'BREEZE'
CHITTERING = 'CHITTERING'
BUMP = 'BUMP'
SCREAM = 'SCREAM'
PAIN = 'PAIN'
# Hazards (query variables)
BAT = 'BAT'
PIT = 'PIT'
WUMPUS = 'WUMPUS'

# ## Rational Agent ##
class RationalAgent(Agent):
  _memo_choose = {}

  def __init__(self):
    pass

  # ### Counting Methods ###

  # Memoized implementation of (n choose k)
  def choose(self, n, k):
    if n < 0: raise ValueError("Received negative n: %d" % n)
    if k < 0: raise ValueError("Received negative k: %d" % n)
    if k > n: raise ValueError("Received k > n: %d, %d" % (k, n))
    if k == 0 or k == n:
      return 1
    elif (n, k) in self._memo_choose:
      return self._memo_choose[(n, k)]
    else:
      return self._memo_choose.setdefault((n, k), self.choose(n-1, k-1) + self.choose(n-1, k))


  # Recursive implementation of config enumeration. Takes a list of
  # "indices" to configure, and a minimum and maximum allowable size
  # for the returned configurations. Outputs an iterator over all
  # configurations of the appropriate size, represented as lists of
  # "indices." Each configuration will appear exactly once in the
  # iteration, as long as the "indices" are unique.
  def configs(self, indices, min_count, max_count):
    if max_count < 0:
      yield []
    if max_count and indices:
      last = len(indices) - min_count + 1
      for i, ii in enumerate(indices[:last]):
        for j in self.configs(indices[i+1:], min_count-1, max_count-1):
          yield [ii] + j
    if min_count <= 0:
      yield []


  # ### Navigation Methods ###

  def reachable(self, room, known_world):
    # Just return every room we can find a path to
    return set(self.safe_paths(room, known_world))


  def safe_rooms(self, known_world):
    # Output a cached result if possible
    cache = known_world.get(self.safe_rooms, (-1, None))
    if cache[0] == known_world.move_num():
      return cache[1].copy()

    # Otherwise, compute which rooms are safe using the inference methods
    safe = set(r for r in set(known_world.visited_rooms()) | known_world.fringe_rooms()
            if self.bat_prob(r, known_world) == 0
              and self.pit_prob(r, known_world) == 0
              and self.wumpus_prob(r, known_world) == 0)

    # Cache and return the results
    known_world[self.safe_rooms] = (known_world.move_num(), safe)
    return safe


  def safe_paths(self, room, known_world):
    # Output a cached result if possible
    cache = known_world.get(self.safe_paths, (-1, None))
    if cache[0] == known_world.move_num() and room in cache[1]:
      return cache[1][room].copy()

    # Otherwise, find the shortest safe path to every room which can be safely reached,
    # using Dijkstra's algorithm
    safe = self.safe_rooms(known_world)
    paths = dict([(room, [])])
    level = set((room, n, d) for d, (n, _) in known_world.neighbors(room).iteritems() if n != room)
    while level:
      paths.update((n, paths[r] + [d]) for r, n, d in level)
      level = set((r, n, d) for _, r, _ in level if r in safe
                  for d, (n, _) in known_world.neighbors(r).iteritems() if n not in paths)

    # Cache and return the results
    known_world.setdefault(self.safe_paths, (known_world.move_num(), {}))[1].setdefault(room, paths)
    return paths.copy()


  # ## Inference Methods ##

  # #### Senses ####
  #
  # In rooms adjacent to or containing a Pit, the Agent perceives a Breeze
  # with probability 1. In rooms containing (but not adjacent to) a Bat, the
  # Agent perceives Chittering with probability 1.
  #
  # In rooms adjacent to or containing a Wumpus, the agent perceives a Stench
  # with probability S1 if there is no Breeze and probability S2 if there is a
  # Breeze. In rooms not containing or adjacent to a Wumpus, the Agent
  # perceives a Stench with probability S3 if there is no Breeze, and
  # probability S4 if there is a Breeze. The locations of stenches are fixed
  # at the beginning of the game, so re-entering a room or killing a Wumpus
  # won't change the Stench of a room.

  # #### KnownWorld info #####
  # `Senses = enum(STENCH=1, BREEZE=2, CHITTERING=4, BUMP=8, SCREAM=16, PAIN=32)`
  #
  #     connectivity()
  #     num_rooms()
  #     num_wumpii()
  #     num_pits()
  #     num_bats()
  #     num_arrows()
  #     stench_prob()
  #     bat_prob()
  #     neighbors()
  #     current_room()
  #     move_num()
  #     shots()
  #     visited_rooms()
  #     fringe_rooms()
  #     adjective()
  #     description()

  def neighbor_set(self, known_world, room):
    return set(n for d, (n, _) in known_world.neighbors(room).iteritems() if n != room)

  def visited_set(self, known_world):
    return set(r for r, sense in known_world.visited_rooms().iteritems())

  # ### bat_prob ###
  # Returns a map from known room numbers to the probability that each room
  # contains a Bat, given the information encoded in `known_world`
  #
  # Remember to consider Bats you've already seen within the maze, and whether
  # or not you've visited each room
  @cached_prob
  def bat_prob(self, known_world):

    result = dict()

    # Number of bats seen is just the sum of the rooms we've seen with
    # CHITTERING.
    for room, sense_code in known_world.visited_rooms().iteritems():
      result[room] = sense_code & Senses.CHITTERING
    num_bats_seen = sum(result.itervalues())

    # Probability of finding a bat in an unvisited room is `number of bats in
    # unvisited rooms / number of unvisited rooms` which is `(number of bats -
    # number of seen bats) / (number of rooms - number of visited rooms)`
    bat_prob = (known_world.num_bats() - num_bats_seen) / (known_world.num_rooms() - len(known_world.visited_rooms()))

    for room in known_world.fringe_rooms():
      result[room] = bat_prob

    pprint("  Bat result:")
    pprint(' ',result)

    return result


  # ### pit_prob(`self, known_world`) ####
  # Returns a map from known room numbers to the probability that each room
  # contains a Pit, given the information encoded in `known_world`
  #
  # > Remember to consider whether or not you've visited each room and whether
  # > or not a particular configuration of Pits yields the pattern of BREEZEs
  # > that you've observed in `known_world`"
  #
  # Expression for probability of pit in query room given `visited_rooms` and
  # BREEZES:
  #
  #     P(pit_query | breezes, visited_rooms) =
  #       \alpha * P(pit_query) * [ \sum_{fringe_rooms} P(breezes | known,
  #       pit_query, pit_fringe_room) * P(pit_fringe_room) ]
  @cached_prob
  def pit_prob(self, known_world):
    # This will hold the mapping to be returned
    result = dict()

    # Shorthand for fringe rooms
    fringe_rooms = known_world.fringe_rooms()

    # If we've visited the room (and we're still alive and doing probability
    # calculations), then it can't contain a pit
    for room in known_world.visited_rooms().keys():
      result[room] = 0

    # The summation for each fringe room starts at 0
    for room in known_world.fringe_rooms():
      result[room] = 0

    # #### Calculate
    #      P(breezes | configuration of pits in fringe, configuration of pits
    #      in other rooms) * P(configuration of pits in fringe) * P(config of
    #      pits in other rooms)
    # #### for each configuration of pits in fringe
    #
    # `P(breezes | configuration of pits in fringe, configuration of pits in
    # other rooms)`
    # is 1 if the breezes are consistent with the configuration of pits

    # ##### is_consistent(`config`) #####
    # Find whether a given pit configuration is consistent with observed
    # breezes. `config` is a list of IDs of rooms with pits.
    # TODO: memoize this
    def is_consistent(config):
      # Generate set of pit-neighbors
      config_neighbors = set()
      for room in config:
        for neighbor in self.neighbor_set(known_world, room):
          config_neighbors.add(neighbor)
      # Return false if there pit with a non-breezy neighbor
      for neighbor in config_neighbors:
        if not known_world.visited_rooms()[neighbor] & Senses.BREEZE:
          return False
      # Return false if there is a breezy room not next to a pit
      for room, sense in known_world.visited_rooms().iteritems():
        if sense & Senses.BREEZE:
          if room not in config_neighbors:
            return False
      return True

    # Generate a set of rooms that can't contain pits because one of their
    # neighbors doesn't have a breeze
    known_pitsafe_rooms = set()
    for room, sense in known_world.visited_rooms().iteritems():
      if not sense & Senses.BREEZE:
        for d, (n, _) in known_world.neighbors(room).iteritems():
          known_pitsafe_rooms.add(n)

    # The indices for the configurations are the IDs of the fringe rooms, since
    # a configuration is an assignment of pits to fringe rooms.
    indices = []
    for room in fringe_rooms:
      if room not in known_pitsafe_rooms:
        indices += [room]

    # If there are more pits than non-fringe, non-visited rooms, then the
    # difference must be in the fringe. So, the minimum number of pits in the
    # fringe is that difference: `num_other_rooms - num_pits`.
    num_other_rooms = known_world.num_rooms() - len(known_world.visited_rooms()) - len(fringe_rooms)
    min_count = max(0, -(num_other_rooms - known_world.num_pits()))
    # The maximum number of pits in the fringe is just the total number of pits.
    max_count = known_world.num_pits()

    # Now iterate through all possible configurations of pits in the fringe
    # that have at least `min_count` pits, and then iterate through each fringe
    # room, counting the configuration if it's consistent with observed
    # breezes.
    consistent_config_count = 0
    for config in RationalAgent.configs(self, indices, min_count, max_count):
      if is_consistent(config):
        num_other_pits = known_world.num_pits() - len(config)
        config_prob = self.choose(num_other_rooms, num_other_pits)
        consistent_config_count += config_prob
        for fringe_room in config:
          result[fringe_room] += config_prob

    # Divide out by total consistent config counts
    for fringe_room in fringe_rooms:
      result[fringe_room] = result[fringe_room] / consistent_config_count

    return result


  # ### wumpus_prob ###
  # Returns a map from known room numbers to the probability that each room
  # contains a Wumpus, given the information encoded in `known_world`
  #
  # > Remember to consider whether or not you've visited each
  # > room and how likely it is for a particular configuration
  # > of Wumpii to yield the pattern of STENCHes that you've
  # > observed in `known_world`. Don't forget that a BREEZE
  # > changes the probability of a STENCH, that killing a Wumpus
  # > doesn't wipe away its STENCH. Finally, remember to take
  # > into account any arrows that you've fired, and the
  # > results!
  @cached_prob
  def wumpus_prob(self, known_world):

    result = dict()

    # Takes a shot-tuple and returns a tuple: (the ID of the room it was shot
    # into, whether or not that shot tuple killed a wumpus)
    def killshot(shot):
      origin, direction, sense = shot
      for d, (n, d2) in known_world.neighbors(origin).iteritems():
        if d is direction:
          return (n, 1 if sense is Senses.SCREAM else 0)

    # Returns a tuple: (list of wumpus graves, list of rooms known to be safe
    # from shooting, number of remaining wumpii in the maze)
    def shot_info():
      num_killed_wumpii = 0
      cleared_rooms = []
      wumpus_graves = []
      for shot in known_world.shots():
        target, got_one = killshot(shot)
        cleared_rooms += [target]
        if got_one:
          wumpus_graves += [target]
        num_killed_wumpii += got_one
      return (wumpus_graves, cleared_rooms, num_killed_wumpii)

    wumpus_graves, cleared_rooms, num_killed_wumpii = shot_info()

    num_remaining_wumpii = known_world.num_wumpii() - num_killed_wumpii

    # Visited rooms and rooms cleared by arrows are safe
    known_safe = set(cleared_rooms + known_world.visited_rooms().keys())

    # A wumpus can't be in a known safe room
    for room in known_safe:
      result[room] = 0

    # The frontier is the set of rooms on the fringe that aren't known to be safe
    frontier = [room for room in known_world.fringe_rooms() if room not in known_safe]

    # The sum over frontier rooms starts at zero
    for room in frontier:
      result[room] = 0

    # ##### calculate_stench_prob(`config`) #####
    # Calculates the probability of an observed pattern of stenches given the
    # configuration of wumpii
    def calculate_stench_prob(config):
      # Corresponds to `[fake: [still: 0, breezy: 0], real: [still: 0, breezy:
      # 0]]`. Boolean indexing FTW.
      stenches = [[0,0],[0,0]]
      nonstenches = [[0,0],[0,0]]

      for room, sense_code in known_world.visited_rooms().iteritems():
        # Is the room smelly?
        has_stench = int(bool(sense_code & Senses.STENCH))
        # Is the room breezy?
        has_breeze = int(bool(sense_code & Senses.BREEZE))
        # Would a stench be real (given this configuration of wumpii and wumpus graves)?
        is_real = 0
        for neighbor in self.neighbor_set(known_world, room):
          if neighbor in config or neighbor in wumpus_graves:
            is_real = 1
            break
        if room in config or room in wumpus_graves:
          is_real = 1
        if has_stench:
          # Count this stench according to breezy/still and real/fake
          stenches[is_real][has_breeze] += 1
        else:
          nonstenches[is_real][has_breeze] += 1

      return ( (        known_world.stench_prob(0,0)  **    stenches[0][0] *
                        known_world.stench_prob(0,1)  **    stenches[0][1] *
                        known_world.stench_prob(1,0)  **    stenches[1][0] *
                        known_world.stench_prob(1,1)  **    stenches[1][1] ) *
               ( (1.0 - known_world.stench_prob(0,0)) ** nonstenches[0][0] *
                 (1.0 - known_world.stench_prob(0,1)) ** nonstenches[0][1] *
                 (1.0 - known_world.stench_prob(1,0)) ** nonstenches[1][0] *
                 (1.0 - known_world.stench_prob(1,1)) ** nonstenches[1][1] ) )

    # The indices for the configurations are the IDs of the frontier rooms, since
    # a configuration is an assignment of wumpii to frontier rooms.
    indices = frontier
    # If there are more wumpii than non-frontier, non-visited rooms, then the
    # difference must be in the frontier. So, the minimum number of wumpii in the
    # frontier is that difference: `num_other_rooms - num_remaining_wumpii`.
    num_other_rooms = known_world.num_rooms() - len(known_safe) - len(frontier)
    pprint("num_other_rooms: ", num_other_rooms)
    min_count = max(0, -(num_other_rooms - num_remaining_wumpii))
    # The maximum number of wumpii in the frontier is just the number of remaining wumpii.
    max_count = num_remaining_wumpii
    # Now iterate through all possible configurations of wumpii in the frontier
    # that have at least `min_count` wumpii
    consistent_config_count = 0
    for config in RationalAgent.configs(self, indices, min_count, max_count):
      num_other_wumpii = known_world.num_wumpii() - len(config)
      summation_term = self.choose(num_other_rooms, num_other_wumpii) * calculate_stench_prob(config)
      consistent_config_count += summation_term
      for fringe_room in config:
        result[fringe_room] += summation_term

    # Divide out by total consistent config counts
    for fringe_room in frontier:
      result[fringe_room] = result[fringe_room] / consistent_config_count

    return result


# ## HybridAgent ##
class HybridAgent(HumanAgent):
  def __init__(self):
    self.__second_brain = RationalAgent()

  def action(self, known_world, rng):
    neighbors = known_world.neighbors(known_world.current_room())
    print "You intuit the probabilities of each hazard..."
    for d in sorted(neighbors.keys(), key=Directions.key):
      n = neighbors[d][0]
      b = self.__second_brain.bat_prob(n, known_world)
      p = self.__second_brain.pit_prob(n, known_world)
      w = self.__second_brain.wumpus_prob(n, known_world)
      visited = n in known_world.visited_rooms().keys()
      str = "\t [%s] %s: Bat %.4f, Pit %.4f, Wumpus, %.4f" % (n, Directions.key(d), b, p, w)
      if visited:
        str += " (visited)"
      print str

    return HumanAgent.action(self, known_world, rng)


# ## SafeAgent ##
class SafeAgent(RationalAgent):
  def __init__(self, danger):
    self.danger = danger
    RationalAgent.__init__(self)

  def action(self, known_world, rng):
    cache = known_world.setdefault(self.action, {})

    # First, check for an express move
    path = cache.get("express", [])
    if path:
      cache["express"] = path[1:]
      return (Actions.MOVE, path[0])

    # Then, check for a target move
    target = cache.pop("target", None)
    if target is not None:
      return target

    # Otherwise, compute a list of places we might possibly want to go. This will
    # consist of just the reachable fringe rooms, unless there are none, in which
    # case it will be the reachable visited rooms containing Bats
    room = known_world.current_room()
    visited = known_world.visited_rooms()
    reachable = self.reachable(room, known_world)
    fringe = known_world.fringe_rooms() & reachable
    if not fringe:
      fringe = set(r for r, s in visited.iteritems() if s & Senses.CHITTERING and r != room) & reachable
    cutoff = 1.0 - known_world.stench_prob(False, False)

    best = 2, []
    paths = self.safe_paths(room, known_world)
    for f in fringe:
      if f == room:
        raise Exception("Bad!")
      # If we know where a Wumpus is and can reach a vantage point, shoot it!
      if f not in visited and self.wumpus_prob(f, known_world) >= cutoff:
        path = paths[f]
        cache["express"] = path[:-1]
        cache["target"] = (Actions.SHOOT, path[-1])
        return self.action(known_world, rng)

      # Otherwise, see what the danger probability is for entering the room
      danger = self.danger(f, known_world)
      if danger < best[0]:
        best = danger, [f]
      elif danger == best[0]:
        best[1].append(f)

    cache["express"] = paths[rng.choice(best[1])]
    return self.action(known_world, rng)


# ## NaiveSafeAgent ##
class NaiveSafeAgent(SafeAgent):
  def __init__(self):
    SafeAgent.__init__(self, self.danger_prob)

  @cached_prob
  def danger_prob(self, known_world):
    # Remember that different hazards are not mutually exclusive, but they are independent!
    # You may want to check out the Wikipedia article on the Inclusion-Exclusion principle
    result = dict()
    # Unless I'm totally mistaken, this is just
    # `1.0 - P(NO_BAT)*P(NO_PIT)*P(NO_WUMPUS)`
    for room in set().union(known_world.visited_rooms().keys(), known_world.fringe_rooms()):
      b = self.bat_prob(room, known_world)
      p = self.pit_prob(room, known_world)
      w = self.wumpus_prob(room, known_world)
      result[room] = (1.0 - ((1.0-b)*(1.0-p)*(1.0-w)))
    return result


# ## BatSafeAgent ##
class BatSafeAgent(SafeAgent):
  def __init__(self):
    SafeAgent.__init__(self, self.lethal_prob)

  @cached_prob
  def lethal_prob(self, known_world):
    # Remember that a Bat in the first room won't necessarily carry you off, but that
    # subsequent Bats will (Bats don't like other Bats!), so long as you don't get
    # killed first. Also, the probability of death in a random room is simpler, but
    # different from, the probability of death in a fixed room, and changes every time
    # you discover a new room non-lethal room. Finally, landing on a Bat which you've
    # already seen before effectively does nothing, as you just swap Bats and don't
    # learn information about a new room
    return {}


# ## CleverAgent ##
class CleverAgent(RationalAgent):
  def __init__(self):
    RationalAgent.__init__(self)

  @cached_prob
  def danger(self, known_world):
    # Remember that different hazards are not mutually exclusive, but they are independent!
    # You may want to check out the Wikipedia article on the Inclusion-Exclusion principle
    result = dict()
    for room in set().union(known_world.visited_rooms().keys(), known_world.fringe_rooms()):
      b = self.bat_prob(room, known_world)

      M = known_world.bat_prob()
      bat_danger = known_world['bat_danger']

      p = self.pit_prob(room, known_world)
      w = self.wumpus_prob(room, known_world)

      result[room] = (1.0 - ((1.0-(b*bat_danger*M))*(1.0-p)*(1.0-w)))
    return result


  def action(self, known_world, rng):
    # Takes a shot-tuple and returns a tuple: (the ID of the room it was shot
    # into, whether or not that shot tuple killed a wumpus)
    def killshot(shot):
      origin, direction, sense = shot
      for d, (n, d2) in known_world.neighbors(origin).iteritems():
        if d is direction:
          return (n, 1 if sense is Senses.SCREAM else 0)

    # Returns a tuple: (list of wumpus graves, list of rooms known to be safe
    # from shooting, number of remaining wumpii in the maze)
    def shot_info():
      num_killed_wumpii = 0
      cleared_rooms = []
      wumpus_graves = []
      for shot in known_world.shots():
        target, got_one = killshot(shot)
        cleared_rooms += [target]
        if got_one:
          wumpus_graves += [target]
        num_killed_wumpii += got_one
      return (wumpus_graves, cleared_rooms, num_killed_wumpii)

    # Update shot_info if stale
    if 'stale_shot_info' not in known_world:
      wumpus_graves, cleared_rooms, num_killed_wumpii = shot_info()
      known_world['shot_info'] = wumpus_graves, cleared_rooms, num_killed_wumpii
      known_world['stale_shot_info'] = False
    elif known_world['stale_shot_info']:
      wumpus_graves, cleared_rooms, num_killed_wumpii = shot_info()
      known_world['shot_info'] = wumpus_graves, cleared_rooms, num_killed_wumpii
      known_world['stale_shot_info'] = False
    # Otherwise get it from cache
    else:
      wumpus_graves, cleared_rooms, num_killed_wumpii = known_world['shot_info']

    num_remaining_wumpii = known_world.num_wumpii() - num_killed_wumpii
    num_remaining_arrows = known_world.num_arrows() - len(known_world.shots())

    # Estimate of danger of getting moved by a bat
    if 'bat_danger' not in known_world:
      R = known_world.num_rooms()
      w = num_remaining_wumpii / R
      p = known_world.num_pits() / R
      known_world['bat_danger'] = (1 - (1 - w)*(1 - p))

    # Measure of how valuable arrows are
    arrow_tolerance = num_remaining_arrows / num_remaining_wumpii

    cache = known_world.setdefault(self.action, {})

    # First, check for an express move
    path = cache.get("express", [])
    if path:
      cache["express"] = path[1:]
      return (Actions.MOVE, path[0])

    # Then, check for a target move
    target = cache.pop("target", None)
    if target is not None:
      # Refresh shot_info()
      known_world['stale_shot_info'] = True
      return target

    # Otherwise, compute a list of places we might possibly want to go. This will
    # consist of just the reachable fringe rooms, unless there are none, in which
    # case it will be the reachable visited rooms containing Bats
    room = known_world.current_room()
    visited = known_world.visited_rooms()
    reachable = self.reachable(room, known_world)
    fringe = known_world.fringe_rooms() & reachable
    if not fringe:
      fringe = set(r for r, s in visited.iteritems() if s & Senses.CHITTERING and r != room) & reachable
    cutoff = 1.0 - known_world.stench_prob(False, False)
    cutoff = cutoff / arrow_tolerance

    best = 2, []
    paths = self.safe_paths(room, known_world)
    for f in fringe:
      if f == room:
        raise Exception("Bad!")
      # If we know where a Wumpus is and can reach a vantage point, shoot it!
      if f not in visited and self.wumpus_prob(f, known_world) >= cutoff:
        path = paths[f]
        cache["express"] = path[:-1]
        cache["target"] = (Actions.SHOOT, path[-1])
        return self.action(known_world, rng)

      # Otherwise, see what the danger probability is for entering the room
      danger = self.danger(f, known_world)
      if danger < best[0]:
        best = danger, [f]
      elif danger == best[0]:
        best[1].append(f)

    cache["express"] = paths[rng.choice(best[1])]
    return self.action(known_world, rng)

# *Vim Modeline:*
# vim: set foldmethod=indent foldlevel=0
