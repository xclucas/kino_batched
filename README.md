kinodynamic rrt is bad at exploring because we have to randomly sample actions and hope that by 
chance we end up where we want to go. But with batching, we can sample N random actions together
(like a 1000 actions) and take the best one (without taking much longer). So that makes it possible to steer a bit.

If we can get shitty steering, then we can do kino rrt like regular rrt, where we sample a random point and try to go there. Then there is a voronoi bias and also a guarantee of complete exploration. We actually want to throw away bad samples because tree search time is very important -- less and higher quality samples is good

## Experimental results
- bottleneck is sequentialism: you need a certain numbers of tree expansions for good exploration (related to visiblity of space), can't get to goal in one step
- bottleneck is NOT sampling - we can increase the batch size until diminishing returns, we still have plenty of bandwith for better and badder sampling


## TODO
More environments
More benchmark statistics
Bidirectional tree
MAB for tree growth, where nodes are arms

Next:
- adaptive branhcing factor like kinopax

## Install and Run
jax, numpy and matplotlib

`python algo.py` (optional: `--viewopt` debug view)

## More notes
The idea is, parallel operations are free (in terms of time) so we want more of those. And we want a smaller tree (less data transfer) since that's bad for the gpu. So this algorithm aims to spend more compute producing fewer but higher quality (more useful to exploration) points. 

## Tree resampling
So what if we have a hard limit on our tree size and it's reached before a solution is found? 

In kino it's common for states to be clustered in some hard to get out of regions. (a lot more than geometric planners). So there are nodes that aren't really helping much, taking up precious GPU memory! We can just simply remove X% of the tree and rerun the planner on this new tree. Statistically, more clustered nodes will be removed than other nodes. This only works if you know your hardcoded tree size is sufficient to hold a succeeding path (Is this optimal? we'll have to see) 

## Things that didn't work

You might think that for our uniform samples, filtering to keep the ones closer to the tree would be better. I mean, it's more likely there are no obstacles between the sample and the tree node that is trying to connect to it. So the connection would likely succeed (or get pretty close at least). But no! In practice the filtered points are *inside* the tree which is useless. Computing any richer inide/outside information would require some denisty estimation which is very hard.

Another is filtering points in collision. Even if the point is in collision (so the node cannot ever connect to it), it may still pull the node in a novel direction. I think this "optimization" also breaks voronoi bias.
