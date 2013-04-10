
import sys
from time import sleep

import wumpus_agents
from wumpus_world import World, KnownWorld, Contents, Senses, Actions, Directions



# ==========================================================================================
# =============================== Begin Description Functions ==============================
# ==========================================================================================

# Print an English description of the hazard Senses
def print_hazard_senses(sense):
  if sense & Senses.STENCH:
    print "You perceive the rancid STENCH of Wumpus"
  if sense & Senses.BREEZE:
    print "You perceive a cool BREEZE"
  if sense & Senses.CHITTERING:
    print "You perceive the CHITTERING of Bats from above"

# Print an English description of the reaction Senses
def print_reaction_senses(sense):
  if sense & Senses.BUMP:
    print "You perceive a loud BUMP"
  if sense & Senses.SCREAM:
    print "You perceive a bloodcurdling SCREAM"
  if sense & Senses.PAIN:
    print "You perceive unbearable PAIN"




# ==========================================================================================
# ================================= Begin Argument Parsing =================================
# ==========================================================================================

#sys.argv.extend(["BatSafe", "-V", "-R", 50, "-W", 2, "-A", 3, "-P", 4, "-B", 3, "-S1", 0.9, "-S2", 0.7, "-S3", 0.1, "-S4", 0.05])


# Define the acceptable arguments
ints = dict(R="nrooms", W="nwumpii", P="npits", B="nbats", A="narrows", C="connectivity", Y="delay")
floats = dict(M="bat_move", S1="real_calm_stench", S2="real_breeze_stench", S3="false_calm_stench", S4="false_breeze_stench")
strings = dict(T="test_file")

# Grab the agent type
agent_type = "Human"
start = 1
if len(sys.argv) > 1:
  flag = sys.argv[1][1:].upper()
  start = 2
  if flag not in ints and flag not in floats and flag not in strings:
    flag = None
    agent_type = sys.argv[1]

# Parse the flags
inputs = {}
verbose = (agent_type == "Human" or agent_type == "Hybrid")
for i, v in enumerate(sys.argv[start:]):
  if flag is None:
    if not v.startswith("-"):
      raise ValueError("Invalid flag syntax: '%s'" % v)
    flag = v[1:].upper()
    if flag == "V":
      verbose = True
      flag = None
  else:
    if flag in ints:
      inputs[ints[flag]] = int(v)
    elif flag in floats:
      inputs[floats[flag]] = float(v)
    elif flag in strings:
      inputs[strings[flag]] = v
    else:
      raise ValueError("Invalid flag: '%s'" % flag)
    flag = None

# Pop the delay out of the input
delay = int(inputs.pop("delay", 0))



# ==========================================================================================
# ================================== Begin Game Functions ==================================
# ==========================================================================================

def describe(world, path, dirs, senses, end=False):
  for i, (r, d, s) in enumerate(zip(path, dirs, senses)):
    limit = (end and i == len(path) - 1)
    print

    # Print the direction you enter from, if applicable
    if d is not None:
      if d & Directions.SOUTH:
        d ^= Directions.NORTH ^ Directions.SOUTH
      if d & Directions.WEST:
        d ^= Directions.EAST ^ Directions.WEST
      dir = Directions.key(d)
      print "You step through the portal and find yourself facing %s" % dir

    # Print the reaction senses and the room description
    if not limit: print_reaction_senses(s)
    print "You are in the %s room. %s" % world.rooms[r]

    if not limit:
      # Print the hazard senses
      print_hazard_senses(s)

      # Print the portals
      print "You see the shimmer of magic portals around the room:"
      neighbors = world.linked_doors[r]
      for d in sorted(neighbors.keys(), key=Directions.key):
        n, d2 = neighbors[d]
        print "\t[%s] %s: %s room, %s side." % (n, Directions.key(d), world.rooms[n][0], Directions.key(d2))

      # Print a bat message, if applicable
      if i < len(path) - 1:
        print "A giant Bat descends from above and carries you off into the darkness..."
        sleep(delay/5000.0)



def run_game():
  # Create a world and an Agent based on the input
  agent = getattr(wumpus_agents, "%sAgent" % agent_type)()
  world = World(**inputs)

  # Create a KnownWorld on top of the World
  handles = {}
  known_world = KnownWorld(world, handles)

  # ============================= Begin Main Action Loop =============================

  # Iterate until death or victory
  path, dirs, senses = [0], [None], [known_world.visited_rooms()[0]]
  wumpii = known_world.num_wumpii()
  arrows = known_world.num_arrows()
  action, dir = None, None
  while not (senses[-1] & Senses.PAIN) and len(known_world.shots()) < known_world.num_arrows() and wumpii > 0:
    # Describe the situation
    if verbose:
      describe(world, path, dirs, senses)

    # Ask the Agent what to do
    sleep(delay/1000.0)
    action, dir = agent.action(known_world, world.agent_rng())
    path, dirs, senses = handles[action](dir)


    # Describe any shots, record any screams
    if action == Actions.SHOOT:
      print "You fire an arrow to the %s, straight and true" % Directions.key(dir)
      if senses[-1] & Senses.SCREAM:
        wumpii -= 1

  # ============================== Begin Final Results ===============================

  # If the last thing we did was move, describe it
  if action == Actions.MOVE:
    describe(world, path, dirs, senses, True)

  # Output the results
  print_reaction_senses(senses[-1] & Senses.BUMP)
  if known_world.shots() and known_world.shots()[-1][2] & Senses.PAIN:
    # Suicide by arrow
    print "Only when you hear a faint *thwump* do you realize that the shot has looped back towards you..."
  elif world.contents[path[-1]] & Contents.WUMPUS:
    # Death by Wumpus
    print "As you enter the room, the stench of Wumpus becomes almost unbearable. You hear a sudden gurgling sound from behind you, but it's already too late..."
  elif world.contents[path[-1]] & Contents.PIT:
    # Death by Pit
    print "The instant you set foot inside the room, the ground opens up beneath you and swallows you whole. The pile of bones cushions your fall, but the sucker marks on the wall make you wish it hadn't..."
  elif wumpii > 0:
    # Out of arrows
    print "Your quiver empty, the hunter becomes the hunted. Desperate to escape, you sprint through the maze, but your flight is cut short by a hideous tentacle..."
    senses[-1] |= Senses.PAIN
  elif known_world.num_wumpii() == 0:
    # No Wumpus!
    print "You descend into the maze, but there's not a Wumpus to be found. I guess it's only a legend after all!"
  else:
    # Victory!
    killed = "final Wumpus" if known_world.num_wumpii() > 1 else "Wumpus"
    pronoun = "those" if known_world.num_wumpii() > 1 else "that"
    plural = "beasts" if known_world.num_wumpii() > 1 else "beast"
    print "Your arrow finds its mark, and slays the %s. At long last, the world is free from %s accursed tentacled %s!" % (killed, pronoun, plural)

  print_reaction_senses(senses[-1] & ~Senses.BUMP)
  return not (senses[-1] & Senses.PAIN)







# Run the game function
try:
  run_game()
except Exception as e:
  import traceback
  print traceback.format_exc()

# When running from the GUI on Windows the output disappears as soon as the script finishes
import platform
if platform.system() == "Windows":
  while True: pass
