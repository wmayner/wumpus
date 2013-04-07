
import random





# This is pretty cool. Modified from: http://stackoverflow.com/questions/36932/whats-the-best-way-to-implement-an-enum-in-python
def enum(*sequential, **named):
	enums = dict(zip(sequential, range(len(sequential))), **named)
	rev = dict((v, k) for k, v in enums.iteritems())
	class Enum:
		class __metaclass__(type):
			def __iter__(self):
				for i in enums.iteritems():
					yield i
			def key(self, val):
				return rev[val]
	for k, v in enums.iteritems():
		setattr(Enum, k, v)
	return Enum



# Set up all the enums we need for the game
Actions = enum('MOVE', 'SHOOT')
Directions = enum(NORTH=12, SOUTH=6, EAST=3, WEST=9, NORTHEAST=1, NORTHWEST=11, SOUTHEAST=5, SOUTHWEST=7)
Contents = enum(WUMPUS=1, PIT=2, BAT=4, AGENT=8)
Senses = enum(STENCH=1, BREEZE=2, CHITTERING=4, BUMP=8, SCREAM=16, PAIN=32)











class World:

	def __init__(self, nrooms=20, nwumpii=1, nbats=0, npits=3, narrows=1,
				connectivity=4, bat_move=0.5,
				real_calm_stench=1, real_breeze_stench=1,
				false_calm_stench=0, false_breeze_stench=0,
				test_file=None):

		# Save the World parameters
		self.nrooms = int(nrooms)
		self.nwumpii = int(nwumpii)
		self.nbats = int(nbats)
		self.npits = int(npits)
		self.narrows = int(narrows)
		self.connectivity = int(connectivity)
		self.bat_move = float(bat_move)
		self.real_calm_stench = float(real_calm_stench)
		self.real_breeze_stench = float(real_breeze_stench)
		self.false_calm_stench = float(false_calm_stench)
		self.false_breeze_stench = float(false_breeze_stench)

		# ============================== Begin RNG Seed Setup ==================================

		# Check for the existence of the test file
		if test_file:
			try:
				recreate = open(test_file, 'r')
			except IOError:
				recreate = None

		# Load seeds from the test file, or generate new ones
		if test_file and recreate:
			internal_seed = int(recreate.readline())
			agent_seed = int(recreate.readline())
			recreate.close()
		else:
			internal_seed = random.randint(0, 2**32 - 1)
			agent_seed = random.randint(0, 2**32 - 1)

		# Write the seeds into a test file, if one was given
		if test_file and not recreate:
			preserve = open(test_file, 'w')
			preserve.write("%d\n" % internal_seed)
			preserve.write("%d\n" % agent_seed)
			preserve.close()

		# Seed the RNG with the internal seed, for use in generating the map
		random.seed(internal_seed)

		# ========================= Begin Room Description Generation ==========================

		# Parse the rooms file
		file = open("adjectives.txt", 'r')
		rooms = []
		rooms = [tuple(([s.strip() for s in line.split(",", 1)]+[""])[:2]) for line in file.readlines() if line.strip()]
		file.close()

		# Select the room descriptions to use (using the internal RNG)
		if self.nrooms > len(rooms):
			raise ValueError("Can't create that many rooms")
		random.shuffle(rooms)
		self.rooms = rooms[:nrooms]

		# ============================ Begin Room Graph Generation =============================

		# Generate the doors
		dirs = set(enum for dir, enum in Directions)
		free_doors = dict((i, dirs.copy()) for i in range(nrooms))
		self.linked_doors = dict((i, {}) for i in range(nrooms))

		# Set up the linking code
		components = [[i] for i in range(nrooms)]
		def link(c1, c2):
			r1 = r2 = -1

			# Must choose and remove door first, in case this is a self-link
			while not free_doors.get(r1):
				r1 = random.choice(components[c1])
			d1 = random.choice(list(free_doors[r1]))
			free_doors[r1].remove(d1)

			while not free_doors.get(r2):
				r2 = random.choice(components[c2])
			d2 = random.choice(list(free_doors[r2]))
			free_doors[r2].remove(d2)

			self.linked_doors[r1][d1] = (r2, d2)
			self.linked_doors[r2][d2] = (r1, d1)

		# First, connect the graph
		while len(components) > 1:
			# Make sure c2 is distinct from c1
			c1 = random.randint(0, len(components)-1)
			c2 = random.randint(0, len(components)-2)
			if c2 >= c1: c2 += 1
			link(c1, c2)
			components[c1].extend(components.pop(c2))

		# Then add more links until the average number of links per room is exactly "connectivity"
		if (nrooms * connectivity) % 2 != 0:
			raise ValueError("Either nrooms or connectivity must be even!")
		for i in xrange(nrooms*connectivity/2 - (nrooms-1)):
			link(0, 0)

		# =============================== Begin Object Placement ===============================

		# Distribute the Wumpii, Pits, and Bats into the other rooms
		if nrooms - 1 < nwumpii:
			raise ValueError("Too many Wumpii in not enough rooms!")
		if nrooms - 1 < npits:
			raise ValueError("Too many Pits in not enough rooms!")
		if nrooms - 1 < nbats:
			raise ValueError("Too many Bats in not enough rooms!")

		# Hazards can go in any room except for room 0, where the agent starts out
		self.contents = [0] * nrooms
		self.contents[0] |= Contents.AGENT
		room_numbers = [r for r in xrange(1, nrooms)]

		# Place the Wumpii
		random.shuffle(room_numbers)
		for i in xrange(nwumpii):
			self.contents[room_numbers[i]] |= Contents.WUMPUS

		# Place the Pits
		random.shuffle(room_numbers)
		for i in xrange(npits):
			self.contents[room_numbers[i]] |= Contents.PIT

		# Place the Bats
		random.shuffle(room_numbers)
		for i in xrange(nbats):
			self.contents[room_numbers[i]] |= Contents.BAT

		# ================================ Begin Sense Placement ===============================

		# Place BREEZES and CHITTERING on top of PITs and BATs
		self.senses = [c & (Senses.BREEZE | Senses.CHITTERING) for c in self.contents]

		# Find the rooms adjacent to Wumpii and Pits
		false_stench = set(xrange(nrooms))
		real_stench = set()
		for i, c in enumerate(self.contents):
			if c & Contents.WUMPUS:
				real_stench.add(i)
				for j, d in self.linked_doors[i].itervalues():
					real_stench.add(j)
			if c & Contents.PIT:
				for j, d in self.linked_doors[i].itervalues():
					self.senses[j] |= Senses.BREEZE

		# Place the STENCHes
		for i, s in enumerate(self.senses):
			if i in real_stench:
				prob = real_breeze_stench if s & Senses.BREEZE else real_calm_stench
			elif i in false_stench:
				prob = false_breeze_stench if s & Senses.BREEZE else false_calm_stench
			if random.uniform(0, 1) < prob:
				self.senses[i] |= Senses.STENCH

		# =============================== Begin RNG Finalization ===============================

		# Finish setting up the RNGs
		self.internal_state = random.getstate()
		random.seed(agent_seed)
		self.agent_state = random.getstate()
		self.rng_for_agent = True

		for i in xrange(nrooms):
			print "Room %d:" % i
			print self.rooms[i]
			print self.contents[i]
			print self.senses[i]
			print self.linked_doors[i]
			print


	# ==========================================================================================
	# ============================= Begin RNG Management Functions =============================
	# ==========================================================================================

	def agent_rng(self):
		# Swap to the agent RNG state if necessary, then return random
		if not self.rng_for_agent:
			self.internal_state = random.getstate()
			random.setstate(self.agent_state)
			self.rng_for_agent = True
		return random


	def internal_rng(self):
		# Swap to the internal RNG state if necessary, then return random
		if self.rng_for_agent:
			self.agent_state = random.getstate()
			random.setstate(self.internal_state)
			self.rng_for_agent = False
		return random


	# ==========================================================================================
	# =============================== Begin Navigation Functions ===============================
	# ==========================================================================================

	def move(self, room, dir):
		new_room, side = self.linked_doors[room].get(dir, (None, None))

		# BUMP if there is no portal in that direction
		if new_room is None:
			return [room], [None], [self.senses[room] | Senses.BUMP]

		# Remove the Agent from the current room, and enter the new room
		self.contents[room] ^= Contents.AGENT
		path, senses = self.__enter(new_room)

		return path, [side] + [None]*len(path), senses


	def __enter(self, room):
		# Place the Agent in the room
		self.contents[room] |= Contents.AGENT
		content = self.contents[room]

		# Start the Agent's path, including PAIN if there is a lethal hazard here
		path, senses = [room], [self.senses[room]]
		if content & (Contents.WUMPUS | Contents.PIT):
			senses[-1] |= Senses.PAIN

		# Loop until the Agent is dead, or not on a bat, as long as the first bat was disturbed
		move = self.internal_rng().uniform(0, 1)
		while (move < self.bat_move) and (content & Contents.BAT) and not (senses[-1] & Senses.PAIN):
			# Remove the Agent, the Bat, and the CHITTERING from the room
			self.contents[room] ^= Contents.AGENT
			self.contents[room] ^= Contents.BAT
			self.senses[room] ^= Senses.CHITTERING

			# Drop the Agent and the Bat in a random room
			room = random.randint(0, self.nrooms-1)
			content = self.contents[room]
			path.append(room), senses.append(self.senses[room] | Senses.CHITTERING)

			# Add PAIN if the Agent landed on a lethal hazard
			if content & (Contents.WUMPUS | Contents.PIT):
				senses[-1] |= Senses.PAIN

			# Add the Agent to the room and toggle the Bat and CHITTERING
			# If there is already a Bat, the next iteration will turn it back on
			# Otherwise, these lines will turn on the Bat and CHITTERING
			self.contents[room] |= Contents.AGENT
			self.senses[room] ^= Senses.CHITTERING
			self.contents[room] ^= Contents.BAT

		return path, senses


	def shoot(self, room, dir):
		target, _ = self.linked_doors[room].get(dir, (None, None))

		# BUMP if there is no portal in that direction
		if target is None:
			return self.senses[room] | Senses.BUMP

		# PAIN if the Agent suicides
		if target == room:
			return self.senses[room] | Senses.PAIN

		# SCREAM if a Wumpus is killed
		if self.contents[target] & Contents.WUMPUS:
			self.contents[target] ^= Contents.WUMPUS
			return self.senses[room] | Senses.SCREAM

		# Nothing on a miss
		return self.senses[room]










class KnownWorld(dict):

	def __init__(self, world, handles):
		position = [0]
		moves = [0]
		shots = []
		visited = set([0])
		fringe = set(n for n, d in world.linked_doors[0].itervalues() if n != 0)

		# ======================= These are the utility methods of KnownWorld =======================
		# They are defined this way in order to make the world itself untouchable by outside code, even
		# by someone who knows about python name mangling. Since we'll be running your code using this
		# implementation of KnownWorld...no peeking!

		# Returns the average number of portals per room
		self.connectivity = lambda: world.connectivity

		# Returns the number of rooms in the maze
		self.num_rooms = lambda: world.nrooms

		# Returns the original number of Wumpii in the maze
		self.num_wumpii = lambda: world.nwumpii

		# Returns the number of Pits in the maze
		self.num_pits = lambda: world.npits

		# Returns the number of Bats in the maze
		self.num_bats = lambda: world.nbats

		# Returns the number of arrows the Agent began with
		self.num_arrows = lambda: world.narrows

		# Returns the probability of perceiving a stench, given whether or not it is real and whether or not there's a breeze
		self.stench_prob = lambda real, breeze: (world.real_breeze_stench if breeze else world.real_calm_stench) if real else (world.false_breeze_stench if breeze else world.false_calm_stench)

		# Returns the probability of disturbing a Bat when entering its room
		self.bat_prob = lambda: world.bat_move

		# Returns the known neighbors of "room," as a map from portal direction in "room"
		# to tuples of (neighbor room number, neighbor portal direction)
		self.neighbors = lambda room: dict((d, (n, d2)) for d, (n, d2) in world.linked_doors[room].iteritems() if n in visited or room in visited)

		# Returns the current room number
		self.current_room = lambda: position[0]

		# Returns the number of moves made so far
		self.move_num = lambda: moves[0]

		# Returns a record of the shots fired, as a tuple of (room, direction, sense), where sense
		# includes BUMPs, SCREAMs, and PAIN
		self.shots = lambda: shots[:]

		# Returns a map from visited room numbers to sensory information
		self.visited_rooms = lambda: dict((r, world.senses[r]) for r in visited)

		# Returns a set of rooms known to exist, but which have not yet been visited
		self.fringe_rooms = lambda: fringe.copy()

		self.adjective = lambda room: world.rooms[room][0] if room in visited or room in fringe else None
		self.description = lambda room: world.rooms[room][1] if room in visited or room in fringe else None



		# ========================= These are action handles for KnownWorld =========================
		# They are used by the internal Wumpus implementation only, so you don't need to worry about
		# them. In fact, you can't even access them from within your Agent!

		def move(dir):
			moves[0] += 1
			path, dirs, senses = world.move(position[0], dir)
			visited.update(path)
			fringe.update(set(n for r in path for n, d in world.linked_doors[r].itervalues()))
			fringe.difference_update(set(visited))
			position[0] = path[-1]
			return path, dirs, senses
		handles[Actions.MOVE] = move

		def shoot(dir):
			moves[0] += 1
			senses = world.shoot(position[0], dir)
			shots.append((position[0], dir, senses & (Senses.BUMP | Senses.SCREAM | Senses.PAIN)))
			return [position[0]], [None], [senses]
		handles[Actions.SHOOT] = shoot
