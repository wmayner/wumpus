from __future__ import division
from operator import iand, mul
from wumpus_world import Contents, Senses, Actions, Directions


class Agent:
  def action(self, known_world, rng):
    raise NotImplementedError()

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


#################################
# Bayesian Network support code #
#############################################################################


# Global constants for variable categories
#==========================================
# Evidence Variables:
STENCH = 'STENCH'
BREEZE = 'BREEZE'
CHITTERING = 'CHITTERING'
BUMP = 'BUMP'
SCREAM = 'SCREAM'
PAIN = 'PAIN'
# Query Variables:
BAT = 'BAT'
PIT = 'PIT'
WUMPUS = 'WUMPUS'


class Node:
  def __init__(self, name, children, parents, data):
    self.name = name
    self.children = children
    self.parents = parents
    self.data = data

  def __repr__(self):
    s = "\nNODE\n----------------------------\n  Name:     %s\n  Children: %s\n  Parents:  %s\n  Data:     %s\n----------------------------\n" % (self.name, self.children, self.parents, self.data)
    return s


class DAG:
  def __init__(self):
    self.nodes = set()


class BN:
  # Implementation of a Bayesian Network tailored to the Wumpus World
  #
  # Defintions:
  # - a 'variable' is a tuple (<category>, <room number>), e.g.
  #   (CHITTERING, 7) or (BAT, 2)
  # - an 'assignment' is a dict that maps variables to values, e.g.
  #   { (CHITTERING, 7): True, (BAT, 2): False }

  def __init__(self, known_world):
    self.dag = DAG()

    # Query Variables:
    self.bat = dict()
    self.pit = dict()
    self.wumpus = dict()
    # Evidence Variables:
    self.chitter = dict()
    self.breeze = dict()
    self.stench = dict()

    ####################
    # Populate Network #
    ##########################################################################
    for room in known_world.visited_rooms():
      neighbors = known_world.neighbors(room)

      # Chitter Node
      #=======================================================================
      name = (CHITTERING, room)
      # The only thing that CHITTERING causes is BAT in the same room, so
      # BAT in this room is the only child
      children = [(BAT, room)]
      parents = None
      # Chittering will never be a query variable, so don't bother with a CPT
      cpt = None
      node = Node(name, children, parents, cpt)
      # add to dag and chitter dict
      self.dag.nodes.add(node)
      self.chitter[room] = node

      # Bat Node
      #=======================================================================
      name = (BAT, room)
      children = None
      # The only thing that causes BAT is CHITTERING in the same room, so
      # CHITTERING in this room is the only parent
      parents = [(CHITTERING, room)]
      # Generate CPT
      #   The CPT is in fact a function that takes an assignment and returns a
      #   list of probabilites, one for each possible value of the variable.
      #   NOTE: Here I'm using a trick that relies on the fact that the
      #   probability of BAT in this room is 1 if we hear CHITTERING in this
      #   room and 0  otherwise.
      cpt = lambda assignment: 1 if all(assignment[variable] for variable in assignment) else 0
      # add to dag and bat dict
      self.dag.nodes.add(node)
      self.bat[room] = node

  def query(variables, evidence):
    # Implementation of TODO: choose inference algorithm
    # Takes a list of variables and an assignment.
    # Returns the posterior conditional probability of each possible value of
    # the set of variables, given the evidence (a dict that maps possible
    # values to probabilities).
    return


##############################################################################


def parse_senses(sense_info):
  # Takes a decimal number representing sensory information
  # Returns a dict mapping senses to booleans indicating their presence
  #
  # Senses = enum(STENCH=1, BREEZE=2, CHITTERING=4, BUMP=8, SCREAM=16, PAIN=32)

  # Convert sense info to binary, split on 'b' to get just the bit string, add
  # leading zeros so there's a digit for each sense, reverse it
  sense_info = bin(sense_info).split('b')[1].zfill(6)[::-1]

  senses = dict()
  for sense, digit in [(STENCH, 0), (BREEZE, 1), (CHITTERING, 2), (BUMP, 3), (SCREAM, 4), (PAIN, 5)]:
    senses[sense] = int(sense_info[digit])
  return senses


##############################################################################

class RationalAgent(Agent):
  _memo_choose = {}

  def __init__(self):
    pass

  # ==========================================================================================
  # ================================= Begin Counting Methods =================================
  # ==========================================================================================

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
    if max_count and indices:
      last = len(indices) - min_count + 1
      for i, ii in enumerate(indices[:last]):
        for j in self.configs(indices[i+1:], min_count-1, max_count-1):
          yield [ii] + j
    if min_count <= 0:
      yield []


  # ==========================================================================================
  # ================================ Begin Navigation Methods ================================
  # ==========================================================================================

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


  # ==========================================================================================
  # ================================ Begin Inference Methods =================================
  # ==========================================================================================

  ##########
  # Senses #
  ############################################################################
  # In rooms adjacent to or containing a Pit, the Agent perceives a Breeze
  # with probability 1. In rooms containing (but not adjacent to) a Bat, the
  # Agent perceives Chittering with probability 1.

  # In rooms adjacent to or containing a Wumpus, the agent perceives a Stench
  # with probability S1 if there is no Breeze and probability S2 if there is a
  # Breeze. In rooms not containing or adjacent to a Wumpus, the Agent
  # perceives a Stench with probability S3 if there is no Breeze, and
  # probability S4 if there is a Breeze. The locations of stenches are fixed
  # at the beginning of the game, so re-entering a room or killing a Wumpus
  # won't change the Stench of a room.

  ###################
  # KnownWorld info #
  ############################################################################
  # Senses = enum(STENCH=1, BREEZE=2, CHITTERING=4, BUMP=8, SCREAM=16, PAIN=32)
  #
  # connectivity()
  # num_rooms()
  # num_wumpii()
  # num_pits()
  # num_bats()
  # num_arrows()
  # stench_prob()
  # bat_prob()
  # neighbors()
  # current_room()
  # move_num()
  # shots()
  # visited_rooms()
  # fringe_rooms()
  # adjective()
  # description()


  @cached_prob
  def bat_prob(self, known_world):
    # Returns a map from known room numbers to the probability that each room
    # contains a Bat, given the information encoded in known_world
    #
    # Remember to consider Bats you've already seen within the maze, and whether
    # or not you've visited each room

    # print '\n=================================================================='
    # print 'known_world'
    # print '------------------------------------------------------------------'
    # print 'Current room:'
    # print known_world.current_room()
    # print 'Neighbors:'
    # print known_world.neighbors(known_world.current_room())
    # print 'Visited rooms:'
    # print known_world.visited_rooms()
    # print 'Fringe rooms:'
    # print known_world.fringe_rooms()
    # print 'Parse senses:'
    # for key, val in known_world.visited_rooms().iteritems():
    #   print parse_senses(val)
    # print
    print '\n=================================================================='
    print 'bat_prob'
    print '------------------------------------------------------------------'

    result = dict()

    # print "### KNOWN ###"
    for room, sense_code in known_world.visited_rooms().iteritems():
      # print "  Room =",room
      result[room] = parse_senses(sense_code)[CHITTERING]

    # Number of bats seen is just the sum of the probabilities of seeing a bat
    # over the known rooms.
    num_bats_seen = sum(result.itervalues())

    # Probability of finding a bat in an unvisited room is
    # number of bats in unvisited rooms / number of unvisited rooms
    #   = (number of bats - number of seen bats) / (number of rooms - number of visited rooms)
    bat_prior_prob = (known_world.num_bats() - num_bats_seen) / (known_world.num_rooms() - len(known_world.visited_rooms()))
    print "  Calculating bat prior:"
    print "  (", str(known_world.num_bats()), '-', str(num_bats_seen), ')  /  (', str((known_world.num_rooms())), '-', str(len(known_world.visited_rooms())), ')  = ', bat_prior_prob

    # print "### FRINGE ###"
    for room in known_world.fringe_rooms():
      # print "  Room =",room
      result[room] = bat_prior_prob

    print "  Bat result:"
    print ' ',result

    return result


  @cached_prob
  def pit_prob(self, known_world):
    # Returns a map from known room numbers to the probability that each room
    # contains a Pit, given the information encoded in known_world
    #
    # Remember to consider whether or not you've visited each room and whether
    # or not a particular configuration of Pits yields the pattern of BREEZEs
    # that you've observed in known_world

    print '------------------------------------------------------------------'
    print 'pit_prob'
    print '------------------------------------------------------------------'

    result = dict()

    print '  Calculating pit prior:'

    # Prior pit probability is just the number of unseen pits / number of unvisited rooms
    pit_prior_prob = known_world.num_pits() / (known_world.num_rooms() - len(known_world.visited_rooms()))
    print "  (", str(known_world.num_pits()), ')  /  (', str((known_world.num_rooms())), '-', str(len(known_world.visited_rooms())), ')  = ', pit_prior_prob

    for room, sense in known_world.visited_rooms().iteritems():
      # If we've visited the room (and we're still alive and doing probability
      # calculations), then it can't contain a pit
      result[room] = 0

    for room in known_world.fringe_rooms():
      result[room] = pit_prior_prob

    print "  Pit result:"
    print ' ',result

    return result


  @cached_prob
  def wumpus_prob(self, known_world):
    # Returns a map from known room numbers to the probability that each room
    # contains a Wumpus, given the information encoded in known_world
    #
    # Remember to consider whether or not you've visited each room and how
    # likely it is for a particular configuration of Wumpii to yield the
    # pattern of STENCHes that you've observed in known_world. Don't forget
    # that a BREEZE changes the probability of a STENCH, that killing a Wumpus
    # doesn't wipe away its STENCH. Finally, remember to take into account any
    # arrows that you've fired, and the results!

    print '------------------------------------------------------------------'
    print 'wumpus_prob'
    print '------------------------------------------------------------------'

    result = dict()

    for room, sense in known_world.visited_rooms().iteritems():
      result[room] = 0

    for room in known_world.fringe_rooms():
      result[room] = 0

    print "  Wumpus result:"
    print ' ',result

    print '==================================================================\n'

    return result


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
      print "\t%s: Bat %.4f, Pit %.4f, Wumpus, %.4f" % (Directions.key(d), b, p, w)

    return HumanAgent.action(self, known_world, rng)


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
      fringe = set(r for r, s in visited.iteritems() if c & Senses.CHITTERING and r != room) & reachable
    cutoff = 1.0 - known_world.stench_prob(False, False)

    best = 1, []
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


class NaiveSafeAgent(SafeAgent):
  def __init__(self):
    SafeAgent.__init__(self, self.danger_prob)

  @cached_prob
  def danger_prob(self, known_world):
    # Remember that different hazards are not mutually exclusive, but they are independent!
    # You may want to check out the Wikipedia article on the Inclusion-Exclusion principle
    return {}


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


class CleverAgent(RationalAgent):
  def action(self, known_world, rng):
    raise NotImplementedError()

# vim: set foldmethod=indent foldlevel=0
