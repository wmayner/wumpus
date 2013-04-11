wumpus
======

This shouldn't need python 2.7 - but if it breaks, please try it with
2.6!

My CleverAgent seems to do well for test cases 1 and 2, but on 3 (the big
one) it takes wayyy to long. I doubt it'll be able to finish 100 trials in
under half an hour. Hopefully I wasn't the only one with this problem and
you guys'll grade us based on the first two test cases XD

### Strategies: ###

When calculating danger of a room, use
`(1 - (bat_prob(room) * M * bat_danger)(1 - pit_prob(room))(1 - wumpus_prob(room)))`
where M is the chance a bat will move you and bat_danger is the following estimate:
`1 - (number of remaining wumpii / number of rooms)*(number of pits / number of rooms)`

Use same cutoff for shooting at a wumpus as NaiveSafeAgent, but divide by
'arrow_tolerance', which is a measure of how valuable arrows are:
`arrows_remaining / wumpii_remaining`

So if we have twice as many arrows as remaining wumpii, then our cutoff will
be halved.
