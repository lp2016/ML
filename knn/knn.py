import heapq
import itertools
import operator
import math
from vis import visualize
from functools import wraps

#这里是原作者的信息，我只是用来学习。。。
__author__ = u'Stefan Kögl <stefan@skoegl.net>'
__version__ = '0.16'
__website__ = 'https://github.com/stefankoegl/kdtree'
__license__ = 'ISC license'


class Node(object):
    """ A Node in a kd-tree

    A tree is represented by its root node, and every node represents
    its subtree"""

    def __init__(self, data=None, left=None, right=None):
        self.data = data
        self.left = left
        self.right = right


    @property
    def children(self):
        """
        Returns an iterator for the non-empty children of the Node

        The children are returned as (Node, pos) tuples where pos is 0 for the
        left subnode and 1 for the right.

        """

        if self.left and self.left.data is not None:
            yield self.left, 0
        if self.right and self.right.data is not None:
            yield self.right, 1


    def set_child(self, index, child):
        """ Sets one of the node's children

        index 0 refers to the left, 1 to the right child """

        if index == 0:
            self.left = child
        else:
            self.right = child


    def height(self):
        """
        Returns height of the (sub)tree, without considering
        empty leaf-nodes

        """

        min_height = int(bool(self))
        return max([min_height] + [c.height()+1 for c, p in self.children])


    def get_child_pos(self, child):
        """ Returns the position if the given child

        If the given node is the left child, 0 is returned. If its the right
        child, 1 is returned. Otherwise None """

        for c, pos in self.children:
            if child == c:
                return pos


    def __repr__(self):
        return '<%(cls)s - %(data)s>' % \
            dict(cls=self.__class__.__name__, data=repr(self.data))


    def __nonzero__(self):
        return self.data is not None

    __bool__ = __nonzero__

    def __eq__(self, other):
        if isinstance(other, tuple):
            return self.data == other
        else:
            return self.data == other.data



def require_axis(f):
    """ Check if the object of the function has axis and sel_axis members """

    @wraps(f)
    def _wrapper(self, *args, **kwargs):
        if None in (self.axis, self.sel_axis):
            raise ValueError('%(func_name) requires the node %(node)s '
                    'to have an axis and a sel_axis function' %
                    dict(func_name=f.__name__, node=repr(self)))

        return f(self, *args, **kwargs)

    return _wrapper



class KDNode(Node):
    """ A Node that contains kd-tree specific data and methods """

    def __init__(self, data=None, left=None, right=None, axis=None,
            sel_axis=None, dimensions=None):
        """ Creates a new node for a kd-tree

        If the node will be used within a tree, the axis and the sel_axis
        function should be supplied.

        sel_axis(axis) is used when creating subnodes of the current node. It
        receives the axis of the parent node and returns the axis of the child
        node. """
        super(KDNode, self).__init__(data, left, right)
        self.axis = axis
        self.sel_axis = sel_axis
        self.dimensions = dimensions


    def axis_dist(self, point, axis):
        """
        Squared distance at the given axis between
        the current Node and the given point
        """
        return math.pow(self.data[axis] - point[axis], 2)


    def dist(self, point):
        """
        Squared distance between the current Node
        and the given point
        """
        r = range(self.dimensions)
        return sum([self.axis_dist(point, i) for i in r])


    def search_knn(self, point, k, dist=None):
        """ Return the k nearest neighbors of point and their distances

        point must be an actual point, not a node.

        k is the number of results to return. The actual results can be less
        (if there aren't more nodes to return) or more in case of equal
        distances.

        dist is a distance function, expecting two points and returning a
        distance value. Distance values can be any comparable type.

        The result is an ordered list of (node, distance) tuples.
        """

        if k < 1:
            raise ValueError("k must be greater than 0.")

        if dist is None:
            get_dist = lambda n: n.dist(point)
        else:
            get_dist = lambda n: dist(n.data, point)

        results = []

        self._search_node(point, k, results, get_dist, itertools.count())

        # We sort the final result by the distance in the tuple
        # (<KdNode>, distance).
        return [(node, -d) for d, _, node in sorted(results, reverse=True)]

    """
    注意了，先按步骤找到叶结点，然后回朔的时候要做两件事，
    1、是更新邻近点，
    2、是检查是否需要检查父结节点的另外一个结点的区域。
    """
    def _search_node(self, point, k, results, get_dist, counter):
        if not self:
            return

        nodeDist = get_dist(self)

        # Add current node to the priority queue if it closer than
        # at least one point in the queue.
        #
        # If the heap is at its capacity, we need to check if the
        # current node is closer than the current farthest node, and if
        # so, replace it.
        item = (-nodeDist, next(counter), self)
        if len(results) >= k:
            if -nodeDist > results[0][0]:
                heapq.heapreplace(results, item)
        else:
            heapq.heappush(results, item)
        # get the splitting plane
        split_plane = self.data[self.axis]


        # Search the side of the splitting plane that the point is in
        if point[self.axis] < split_plane:
            if self.left is not None:
                self.left._search_node(point, k, results, get_dist, counter)
        else:
            if self.right is not None:
                self.right._search_node(point, k, results, get_dist, counter)

        #result中记录了k个与目标点邻近的点，值是这k个邻近点与目标点距离的平方的相反数
        #这里是检查是否需要检查父结节点的另外一个结点的区域，
        # 也就是判断以目标点为圆心，以result中与目标点距离最远的点的距离为半径所形成的圆与父结点的另一个区域相交
        #说白了，就是公式：[ 目标值（按轴读值） - 父节点（按轴读值）] **2  < （圆的半径）
        plane_dist = point[self.axis] - split_plane
        plane_dist2 = plane_dist * plane_dist
        if -plane_dist2 > results[0][0] or len(results) < k:
            if point[self.axis] < self.data[self.axis]:
                if self.right is not None:
                    self.right._search_node(point, k, results, get_dist,
                                            counter)
            else:
                if self.left is not None:
                    self.left._search_node(point, k, results, get_dist,
                                           counter)


    @require_axis
    def search_nn(self, point, dist=None):
        """
        Search the nearest node of the given point

        point must be an actual point, not a node. The nearest node to the
        point is returned. If a location of an actual node is used, the Node
        with this location will be returned (not its neighbor).

        dist is a distance function, expecting two points and returning a
        distance value. Distance values can be any comparable type.

        The result is a (node, distance) tuple.
        """

        return next(iter(self.search_knn(point, 1, dist)), None)




def create(point_list=None, dimensions=None, axis=0, sel_axis=None):
    """ Creates a kd-tree from a list of points
        利用列表中的数创建一个kd树（利用后序遍历，递归）
    All points in the list must be of the same dimensionality.
    所有的点维度必须一致
    If no point_list is given, an empty tree is created. The number of
    dimensions has to be given instead.

    If both a point_list and dimensions are given, the numbers must agree.

    Axis is the axis on which the root-node should split.

    sel_axis(axis) is used when creating subnodes of a node. It receives the
    axis of the parent node and returns the axis of the child node. """

    if not point_list and not dimensions:
        raise ValueError('either point_list or dimensions must be provided')

    elif point_list:
        dimensions = check_dimensionality(point_list, dimensions)

    # by default cycle through the axis
    sel_axis = sel_axis or (lambda prev_axis: (prev_axis+1) % dimensions)

    if not point_list:
        return KDNode(sel_axis=sel_axis, axis=axis, dimensions=dimensions)

    # Sort point list and choose median as pivot element
    point_list = list(point_list)
    point_list.sort(key=lambda point: point[axis])
    median = len(point_list) // 2

    loc   = point_list[median]
    left  = create(point_list[:median], dimensions, sel_axis(axis))
    right = create(point_list[median + 1:], dimensions, sel_axis(axis))
    return KDNode(loc, left, right, axis=axis, sel_axis=sel_axis, dimensions=dimensions)


def check_dimensionality(point_list, dimensions=None):
    dimensions = dimensions or len(point_list[0])
    for p in point_list:
        if len(p) != dimensions:
            raise ValueError('All Points in the point_list must have the same dimensionality')

    return dimensions


if __name__ == '__main__':
    #emptyTree = create(dimensions=3)
    point1=[2,3]
    point2 =[5,4]
    point3 = [9,6]
    point4 = [4,7]
    point5 = [8,1]
    point6 = [7,2]

    point=[point1,point2,point3,point4,point5,point6]
    tree=create(point)
    visualize(tree)                  # checck the tree
    #result=tree.search_nn((1,2))    #查找最近点，实际上是临近点的一种特殊形式，即只有一个邻近点
    result2=tree.search_knn((5,6),2)
    print(result2)

