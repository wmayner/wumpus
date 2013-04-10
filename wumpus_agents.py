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


DEBUG = 'DEBUG DANGER: '
DEBUG = False
def pprint(*stuff):
  if not DEBUG:
    return
  import re
  string = ''
  for thing in stuff:
    string += str(thing)
  if re.compile(r"\d").match(string) == None and DEBUG == 1:
    print string
  elif DEBUG == stuff[0]:
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


#### parse_senses ####
# Takes a decimal number representing sensory information
# Returns a dict mapping senses to booleans indicating their presence
#
# `Senses = enum(STENCH=1, BREEZE=2, CHITTERING=4, BUMP=8, SCREAM=16, PAIN=32)`
#
def parse_senses(sense_info):
  # Convert sense info to binary, split on `'b'` to get just the bit string, add
  # leading zeros so there's a digit for each sense, reverse it
  sense_info = bin(sense_info).split('b')[1].zfill(6)[::-1]

  senses = dict()
  for digit, sense in enumerate([STENCH, BREEZE, CHITTERING, BUMP, SCREAM, PAIN]):
    senses[sense] = int(sense_info[digit])
  return senses


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
    # print '\n=================================================================='
    # print 'known_world'
    # print '------------------------------------------------------------------'
    # print 'Current room:'
    # print known_world.current_room()
    # print 'Neighbors:'
    # print known_world.neighbors(known_world.current_room())
    # print 'Visited rooms:'
    # print known_world.visited_rooms()
    # print '##### My funcs'
    # print 'Neighbor set of current room:'
    # print self.neighbor_set(known_world, known_world.current_room())
    # print 'Visited set:'
    # print self.visited_set(known_world)
    # print '#####'
    # print 'Fringe rooms:'
    # print known_world.fringe_rooms()
    # print 'Parse senses:'
    # for key, val in known_world.visited_rooms().iteritems():
      # print parse_senses(val)
    # print
    pprint('\n==================================================================')
    pprint('bat_prob')
    pprint('------------------------------------------------------------------')

    result = dict()

    # Number of bats seen is just the sum of the rooms we've seen with
    # CHITTERING.
    for room, sense_code in known_world.visited_rooms().iteritems():
      result[room] = parse_senses(sense_code)[CHITTERING]
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
    pprint('------------------------------------------------------------------')
    pprint('pit_prob')
    pprint('------------------------------------------------------------------')

    # This will hold the mapping to be returned
    result = dict()

    # Shorthand for fringe rooms
    fringe_rooms = known_world.fringe_rooms()

    # If we've visited the room (and we're still alive and doing probability
    # calculations), then it can't contain a pit
    for room in self.visited_set(known_world):
      result[room] = 0

    # #### Calculate prior probability of a pit in a fringe room ####
    # This is just the `number of pits / number of
    # known-safe rooms`
    # TODO: cache this
    pit_prior_prob = known_world.num_pits() / (known_world.num_rooms() - 1)

    pprint(known_world.num_pits())
    pprint(known_world.num_rooms())
    pprint(known_world.num_rooms() - len(self.visited_set(known_world)))
    pprint(self.visited_set(known_world))

    pprint('  Calculating pit prior:')
    pprint(pit_prior_prob)

    # #### Calculate
    #      P(breezes | visited_rooms, pit in query, configuration of pits in fringe) * P(configuration of pits in fringe)
    # #### for each room in fringe
    #
    # `P(breezes | visited_rooms, pit in query, configuration of pits in fringe)`
    # is 1 if the breezes are consistent with there being no pits in
    # `visited_rooms`, a pit in the query room, and the configuration of pits
    # we're considering; 0 otherwise.
    #
    # So, the value of the expression above is just the sum over the (prior)
    # probabilities of the configurations of pits in the frontier that are
    # consistent with observed breezes.

    # print '----------------'
    # print 'Fringe rooms:'
    # print fringe_rooms
    # print '----------------'

    # ##### is_consistent(`config`) #####
    # Find whether a given pit configuration is consistent with observed
    # breezes. `config` is a list of IDs of rooms with pits.
    # TODO: memoize this
    def is_consistent(config):
      # print '#### in is_consistent with config: ', config
      result = True

      # Represent the config as a mapping of fringe rooms to whether of not
      # they have pits
      has_pit = dict()
      for room in fringe_rooms:
        has_pit[room] = room in config

      # print ' has_pit: ', has_pit

      # If `room` has a pit in this configuration, check that all its known
      # neighbors have breezes
      for room in fringe_rooms:
        neighbors = self.neighbor_set(known_world, room)
        # print '  room: ', room
        # print '  neighbors: ', neighbors
        if has_pit[room]:
          # print '  room has pit: '
          for neighbor in neighbors:
            if neighbor in self.visited_set(known_world):
              sense_code = known_world.visited_rooms()[neighbor]
              # print '    neighbor:', neighbor
              # print '    sense code:', parse_senses(sense_code)[BREEZE]
              result = result and parse_senses(sense_code)[BREEZE]

      # Check that all rooms with breezes have at least one neighbor with a pit
      # in this configuration
      for room, sense_code in known_world.visited_rooms().iteritems():
        if parse_senses(sense_code)[BREEZE]:
          room_consistent = False
          neighbors = self.neighbor_set(known_world, room)
          for neighbor in neighbors:
            if neighbor in fringe_rooms:
              if has_pit[neighbor]:
                room_consistent = True
          result = result and room_consistent

      # print '  returning:', result
      # print '######################'
      return result

    # ##### prob_of_config_given(`query, config`) #####
    # Returns the (prior) probability of a configuration not including the
    # query room.
    def prob_of_config_given(query, config):
      result = 1
      for room in fringe_rooms:
        # Exclude query room.
        if room is not query:
          # print "      Considering room ", room
          if room in config:
            # There is a pit in `room`, so multiply by prior probability of a
            # pit in a room

            # print "        Pit: Mutliplying result by prior pit: ", pit_prior_prob

            result *= pit_prior_prob
          elif room not in config:
            # There is no pit in `room`, so multiply by prior probability of no
            # pit in a room

            # print "        No Pit: Mutliplying result by (1.0 - prior pit): ", (1.0 - pit_prior_prob)

            result *= (1.0 - pit_prior_prob)

      return result

    # The indices for the configurations are the IDs of the fringe rooms, since
    # a configuration is an assignment of pits to fringe rooms.
    indices = list(fringe_rooms)
    # If there are more pits than non-fringe, non-visited rooms, then the
    # difference must be in the fringe. So, the minimum number of pits in the
    # fringe is that difference: `num_other_rooms - num_pits`.
    num_other_rooms = known_world.num_rooms() - len(known_world.visited_rooms()) - len(fringe_rooms)
    min_count = max(0, -(num_other_rooms - known_world.num_pits()))
    # The maximum number of pits in the fringe is just the total number of pits.
    max_count = known_world.num_pits()

    # The sum over fringe rooms starts at zero for each query room and for each
    # possible value of the query, i.e. PIT and NO PIT
    for query in fringe_rooms:
      result[query] = [0, 0]

    # Constants for indexing into the entries of the result dict
    PROB_OF_PIT = 0
    PROB_OF_NO_PIT = 1

    # Now iterate through all possible configurations of pits in the fringe
    # that have at least `min_count` pits, and then iterate through each fringe
    # room, counting the configuration if it's consistent with observed
    # breezes.
    #
    # Note that we're also multiplying by the prior probability of each
    # configuration. This is just `(the prior probability of a pit in a room) *
    # (the number of pits in the configuration, minus 1 for the query pit)`
    for config in RationalAgent.configs(self, indices, min_count, max_count):

      # Only consider configurations of pits in the fringe that are consistent with observed breezes.
      if is_consistent(config):

        # print "Config: ", [ (room, room in config) for room in fringe_rooms ]

        # Given this configuration of pits, compute the next term in the
        # summation for each fringe room as the query
        for query in fringe_rooms:

          # print "  Query: ", query

          # Add the next term in the summation for this query.  This term is
          # given by `(prior probability of a pit) * (number of rooms in fringe
          # not including query with pit) * (prior probability of no pit) *
          # (number of rooms in fringe not including query with no pit)`
          summation_term = prob_of_config_given(query, config)
          if query in config:
            # print "    Adding to query HAS PIT probability: ",summation_term
            result[query][PROB_OF_PIT] += summation_term
          elif query not in config:
            # print "    Adding to query NO PIT probability: ",summation_term
            result[query][PROB_OF_NO_PIT] += summation_term

    for query in fringe_rooms:
      # Now mutiply each fringe query by the prior probability of the query
      # taking that particular value (pit / no pit)
      result[query][PROB_OF_PIT] *= pit_prior_prob
      result[query][PROB_OF_NO_PIT] *= (1.0 - pit_prior_prob)
      # Calculate and multiply by normalization constant

      # print "query: ", query
      # print "result[query][PROB_OF_PIT] = ",result[query][PROB_OF_PIT]
      # print "result[query][PROB_OF_NO_PIT] = ",result[query][PROB_OF_NO_PIT]

      # This gives the value for `alpha` since
      # ```
      # (sum over all values of query) * alpha = 1
      # ```
      alpha = 1 / (result[query][PROB_OF_PIT] + result[query][PROB_OF_NO_PIT])

      # print " ALPHA: ", alpha

      # Forget the probability of there being no pit in the room, since it's a
      # boolean and we only need one of the two - and we're returning a map
      # only to the probability of a pit, in any case
      result[query] = result[query][PROB_OF_PIT] * alpha

    # print '----------------'

    pprint("  Pit result:")
    pprint(' ',result)

    # Finally, return the mapping
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
    pprint('------------------------------------------------------------------')
    pprint('wumpus_prob')
    pprint('------------------------------------------------------------------')

    pprint("Shots: ",known_world.shots())

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
    pprint("Wumpii remaining: ",num_remaining_wumpii)

    # Visited rooms and rooms cleared by arrows are safe
    known_safe = set(cleared_rooms + known_world.visited_rooms().keys())

    # Prior probability of a Wumpus being in a given room is the `number of
    # remaining wumpii / number of rooms not known to be safe`
    wumpus_prior = num_remaining_wumpii / (known_world.num_rooms() - len(known_safe))
    pprint("wumpus_prior: ",wumpus_prior)

    # A wumpus can't be in a known safe room
    for room in known_safe:
      result[room] = 0

    # The frontier is the set of rooms on the fringe that aren't known to be safe
    frontier = []
    for room in known_world.fringe_rooms():
      if room not in known_safe:
        frontier += [ room ]

    pprint("frontier: ",frontier)

    # The sum over frontier rooms starts at zero for each query room and for each
    # possible value of the query, i.e. WUMPUS and NO_WUMPUS
    for query in frontier:
      result[query] = [0, 0]

    # Constants for indexing into the entries of the result dict
    NO_WUMPUS = 0
    WUMPUS = 1

    # ##### prob_of_config_given(`query, config`) #####
    # Returns the (prior) probability of a configuration not including the
    # query room.
    def prob_of_config_given(query, config):
      pprint("  Query: ",query)
      result = 1
      for room in frontier:
        # Exclude query room.
        if room is not query:
          pprint("      Considering room ", room)
          if room in config:

            pprint("        wumpus: Mutliplying result by wumpus_prior: ", wumpus_prior)

            # There is a wumpus in `room`, so multiply by prior probability of a
            # wumpus in a room
            result *= wumpus_prior
          elif room not in config:

            pprint("        no wumpus: Mutliplying result by (1.0 - wumpus_prior): ", (1.0 - wumpus_prior))

            # There is no pit in `room`, so multiply by prior probability of no
            # pit in a room
            result *= (1.0 - wumpus_prior)
      return result

    # ##### calculate_stench_prob(`known_safe, query, config, frontier`) #####
    # Calculates the probability of an observed pattern of stenches given the
    # wumpus value of the query room, a configuration of wumpii in the
    # frontier, and known safe rooms
    def calculate_stench_prob(known_safe, config, frontier):
      # Corresponds to `[fake: [still: 0, breezy: 0], real: [still: 0, breezy:
      # 0]]`. Boolean indexing FTW.
      stenches = [[0,0],[0,0]]
      nonstenches = [[0,0],[0,0]]

      for room, sense_code in known_world.visited_rooms().iteritems():
        senses = parse_senses(sense_code)

        # Is the room breezy?
        has_breeze = senses[BREEZE]

        # Would a stench be real (given this configuration of wumpii and wumpus graves)?
        is_real = 0
        for neighbor in self.neighbor_set(known_world, room):
          if neighbor in config or neighbor in wumpus_graves:
            is_real = 1
            break
        if room in config or room in wumpus_graves:
          is_real = 1

        if senses[STENCH]:
          # Count this stench according to breezy/still and real/fake
          stenches[is_real][has_breeze] += 1
        elif not senses[STENCH]:
          pprint('nonstench: ',room, ' is_real: ', is_real, ' is_breezy', has_breeze)
          nonstenches[is_real][has_breeze] += 1

      # self.stench_prob = lambda real, breeze: (world.real_breeze_stench if breeze else world.real_calm_stench) if real else (world.false_breeze_stench if breeze else world.false_calm_stench)
      pprint("stenches: ", stenches)
      pprint("stench terms:")
      pprint(known_world.stench_prob(0,0) ,': ', stenches[0][0])
      pprint(known_world.stench_prob(0,1) ,': ', stenches[0][1])
      pprint(known_world.stench_prob(1,0) ,': ', stenches[1][0])
      pprint(known_world.stench_prob(1,1) ,': ', stenches[1][1])
      pprint("nonstenches: ", nonstenches)
      pprint("non-stench terms:")
      pprint((1.0 - known_world.stench_prob(0,0)),': ', nonstenches[0][0])
      pprint((1.0 - known_world.stench_prob(0,1)),': ', nonstenches[0][1])
      pprint((1.0 - known_world.stench_prob(1,0)),': ', nonstenches[1][0])
      pprint((1.0 - known_world.stench_prob(1,1)),': ', nonstenches[1][1])
      return ( (      known_world.stench_prob(0,0)  **    stenches[0][0] *
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
    # min_count = max(0, -(num_other_rooms - num_remaining_wumpii))
    min_count = 0
    pprint("min_count: ", min_count)
    # The maximum number of wumpii in the frontier is just the number of remaining wumpii.
    max_count = num_remaining_wumpii
    # Now iterate through all possible configurations of wumpii in the frontier
    # that have at least `min_count` wumpii, and then iterate through each frontier
    # room, calculating the probability of the pattern of observed stenches
    # given known safe rooms and WUMPUS or NO_WUMPUS in query room
    #
    # Note that we're also multiplying by the prior probability of each
    # configuration. This is just `(the prior probability of a wumpus in a room) *
    # (the number of wumpii in the configuration, not including query) * (the
    # prior probability of no wumpus in a room) * (the number of rooms devoid
    # of wumpii in the configuration, not including query)`
    for config in RationalAgent.configs(self, indices, min_count, max_count):
      pprint("Considering config: ",config)
      # The probability of observed stench pattern given this
      # configuration, and known safe rooms
      stench_prob = calculate_stench_prob(known_safe, config, frontier)

      pprint("   stench prob: ",stench_prob)

      for query in frontier:
        if query in config:
          query_value = WUMPUS
        elif query not in config:
          query_value = NO_WUMPUS
        # Don't bother calculating config_prob if stench_prob is 0
        if stench_prob != 0:
          # The probability of this configuration given the query
          config_prob = prob_of_config_given(query, config)
          pprint("   config_prob: ",config_prob)
        # The summation term for this query value is given by [ conditional
        # probability of observed stenches given known safe rooms, the query,
        # and this configuration (`stench_prob`) ] * [ the probability of this
        # configuration given the query (`config_prob`) ]
          result[query][query_value] += (stench_prob * config_prob)
          pprint("   adding ",(stench_prob*config_prob))
        else:
          result[query][query_value] += 0
        pprint("   result[ ",query," ][ ",query_value,"] is now: ", result[query][query_value])

    # Now mutiply each frontier query by the prior probability of the query
    # taking that particular value (WUMPUS or NO_WUMPUS), then calculate and
    # multiply by the normalization constant, `alpha`
    for query in frontier:
      pprint("pre-prior-multiplied W prob for ",query,": ",result[query][WUMPUS])
      pprint("pre-prior-multiplied NO_W prob for ",query,": ",result[query][NO_WUMPUS])
      result[query][WUMPUS] *= wumpus_prior
      result[query][NO_WUMPUS] *= (1.0 - wumpus_prior)
      pprint("pre-normalized W prob for ",query,": ",result[query][WUMPUS])
      pprint("pre-normalized W prob for ",query,": ",result[query][NO_WUMPUS])

      # This gives the value for `alpha` since
      # ```
      # (sum over all values of query) * alpha = 1
      # ```
      alpha = 1 / (result[query][WUMPUS] + result[query][NO_WUMPUS])
      pprint("Alpha: ",alpha)
      # Forget the probability of there being NO_WUMPUS in the room, since it's
      # a boolean and we only need one of the two - and we're returning a map
      # only to the probability of a pit, in any case
      result[query] = result[query][WUMPUS] * alpha

    pprint("  Wumpus result:")
    pprint(' ',result)

    pprint('==================================================================\n')

    # print parse_senses(known_world.visited_rooms()[known_world.current_room()])

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

    pprint("DEBUG DANGER: ","FRINGE ROOMS: ",known_world.fringe_rooms())

    # Unless I'm totally mistaken, this is just
    # `1.0 - P(NO_BAT)*P(NO_PIT)*P(NO_WUMPUS)`
    for room in set().union(known_world.visited_rooms().keys(), known_world.fringe_rooms()):
      pprint("DEBUG DANGER: ","ROOM: ",room,(" (visited)" if room in known_world.visited_rooms().keys() else ''))
      b = self.bat_prob(room, known_world)
      pprint("DEBUG DANGER: ","\t\tb-prob of ",b)
      p = self.pit_prob(room, known_world)
      pprint("DEBUG DANGER: ","\t\tp-prob of ",p)
      w = self.wumpus_prob(room, known_world)
      pprint("DEBUG DANGER: ","\t\tw-prob of ",w,",  cutoff = ", (1.0 - known_world.stench_prob(False, False)))
      pprint("DEBUG DANGER: ","\t((1-b)*(1-p)*(1-w)) = ",((1.0-b)*(1.0-p)*(1.0-w)))
      result[room] = (1.0 - ((1.0-b)*(1.0-p)*(1.0-w)))
      pprint("DEBUG DANGER: ","\tdanger_prob(",room,"): ",result[room])

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
  def action(self, known_world, rng):
    raise NotImplementedError()

# *Vim Modeline:*
# vim: set foldmethod=indent foldlevel=0
