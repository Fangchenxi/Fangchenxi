package org.example;

import org.junit.Test;
import sun.security.action.PutAllAction;

import java.lang.management.GarbageCollectorMXBean;
import java.util.*;


public class Main {
    public static void main(String[] args) {
        System.out.println("Hello world!");
    }

    List<List<Integer>> res = new ArrayList<>();
    LinkedList<Integer> path = new LinkedList<>();

    /**
     * Description: 求和为target的n个k面骰子的组合数，由于不是求具体的组合，这里用动规更适合
     * date: 2023/10/24 21:30
     *
     * @param n
     * @param k
     * @param target
     * @author: Chenxi Fang
     */

    public int numRollsToTarget(int n, int k, int target) {
        final int MOD = 1000000007;
        int[][] dp = new int[n + 1][target + 1];
        dp[0][0] = 1;
        for (int i = 1; i <= n; i++) {
            for (int j = 1; j <= target; j++) {
                for (int l = 1; l <= k; l++) {
                    if (j - l >= 0) {
                        dp[i][j] = (dp[i][j] + dp[i - 1][j - l]) % MOD;
                    } else break;
                }
            }
        }
        return dp[n][target];
    }

    /**
     * Description: 用回溯会超时
     * date: 2023/10/24 21:36
     *
     * @param target
     * @param count
     * @param k
     * @param n
     * @author: Chenxi Fang
     */

    public void backtracking(int target, int count, int k, int n) {
        if (target == 0) {
            if (count == n) res.add(new ArrayList<>(path));
            return;
        }
        for (int i = 1; i <= k; i++) {
            if (target - i < 0) break;
            path.add(i);
            count++;
            backtracking(target - i, count, k, n);
            count--;
            path.removeLast();
        }
    }

    /**
     * Description: 将字符串转换成z字排列后得到新字符串
     * date: 2023/10/25 20:07
     *
     * @param s
     * @param numRows
     * @author: Chenxi Fang
     */

    public String convert(String s, int numRows) {
        if (numRows == 1) return s;
        int recurLen = 2 * numRows - 2;
        int len = s.length();
        StringBuilder[] rows = new StringBuilder[numRows];
        for (int i = 0; i < numRows; i++) {
            rows[i] = new StringBuilder();
        }
        int a = len / recurLen;
        int b = len % recurLen;
        for (int i = 0; i <= a; i++) {
            int index = numRows - 2;
            int end = i == a ? b : recurLen;
            for (int j = 0; j < end; j++) {
                char c = s.charAt(i * recurLen + j);
                if (j < numRows) {
                    rows[j].append(c);
                } else {
                    rows[index--].append(c);
                }
            }
        }
        StringBuilder res = new StringBuilder();
        for (int i = 0; i < numRows; i++) {
            res.append(rows[i]);
        }
        return res.toString();
    }

    /**
     * Description: 求小于等于n的惩罚数的平方和，惩罚数为平方后按位分割的数组和等于原数，直接遍历加回溯解决
     * date: 2023/10/25 20:45
     *
     * @param n
     * @author: Chenxi Fang
     */
    public int punishmentNumber(int n) {
        int res = 0;
        for (int i = 1; i <= n; i++) {
            if (backtracking(i, String.valueOf(i * i), 0, 0)) {
                res += i * i;
            }
        }
        return res;
    }

    /**
     * Description: 回溯不一定为空返回值，要视题目而定，这里设为boolean很重要
     * date: 2023/10/25 20:43
     *
     * @param n
     * @param num
     * @param index
     * @param sum
     * @author: Chenxi Fang
     */
    public boolean backtracking(int n, String num, int index, int sum) {
        if (index == num.length()) {
            return sum == n;
        }
        int cur;
        for (int i = index; i < num.length(); i++) {
            cur = Integer.parseInt(num.substring(index, i + 1));
            if (cur + sum > n) break;
            if (backtracking(n, num, i + 1, sum + cur)) {
                return true;
            }
        }
        return false;

    }

    /***
     * Description: 冒泡排序
     * date: 2023/10/25 22:24
     * @param arr
     * @author: Chenxi Fang
     */
    public int[] sortArr(int[] arr) {
        for (int i = 0; i < arr.length - 1; i++) {
            for (int j = i; j < arr.length - i - 1; j++) {
                if (arr[j] > arr[j + 1]) {
                    int temp = arr[j];
                    arr[j] = arr[j + 1];
                    arr[j + 1] = temp;
                }
            }
        }
        return arr;
    }


    /***
     * Description: 给定水平和垂直切割的数组，问切割后最大的蛋糕块的大小
     * 不要急着用深搜这种较难实现的，先试下贪心算法或动规，不行再深搜
     * 这里使切割线之间的长度最大，切割后的面积就最大，另外还要考虑边界情况，0和border的特例
     * date: 2023/10/27 23:25
     * @param h
     * @param w
     * @param horizontalCuts
     * @param verticalCuts
     * @author: Chenxi Fang
     */
    public int maxArea(int h, int w, int[] horizontalCuts, int[] verticalCuts) {
        Arrays.sort(horizontalCuts);
        Arrays.sort(verticalCuts);
        return (int) ((long) calMax(horizontalCuts, h) * calMax(verticalCuts, w) % 1000000007);
    }

    public int calMax(int[] arr, int board) {
        int res = 0;
        for (int i = 1; i < arr.length; i++) {
            res = Math.max(res, arr[i] - arr[i - 1]);
        }
        return Math.max(Math.max(res, arr[0]), board - arr[arr.length - 1]);
    }

    public long pickGifts(int[] gifts, int k) {
        if (gifts.length == 0) return 0;
        PriorityQueue<Integer> queue = new PriorityQueue<>(((o1, o2) -> o2 - o1));
        for (int gift : gifts) {
            queue.offer(gift);
        }
        long res = 0;
        for (int i = 0; i < k; i++) {
            int maxGift = queue.poll();
            queue.offer((int) Math.sqrt(maxGift));
        }
        while (!queue.isEmpty()) res += queue.poll();
        return res;
    }


    /**
     * Description: 将字符串转化成数字，因为要考虑的因素较多，为防止代码臃肿，构造自动机来解决
     * date: 2023/10/28 23:38
     *
     * @param str
     * @author: Chenxi Fang
     */

    public int myAtoi(String str) {
        Automation automation = new Automation();
        for (int i = 0; i < str.length(); i++) {
            automation.get(str.charAt(i));
        }
        return (int) (automation.sign * automation.ans);
    }

    public int hIndex(int[] citations) {
        int[] array = Arrays.stream(citations).boxed().sorted(((o1, o2) -> o2 - o1)).mapToInt(Integer::intValue).toArray();
        int res = 0;
        for (int i = 0; i < array.length; i++) {
            if (array[i] >= i + 1) {
                res = i + 1;
            } else {
                break;
            }
        }
        return res;
    }

    public int hIndex2(int[] citations) {
        int n = citations.length;
        int left = 0, right = n - 1;
        while (left <= right) {
            int mid = left + (right - left) / 2;
            if (n - mid <= citations[mid]) {
                right = mid - 1;
            } else {
                left = mid + 1;
            }
        }
        return n - left;
    }

    /**
     * Description: 0-(n-1) 的各节点，求每个节点为根的树所不包含的最小值，nums是各节点的值，各不相同且都大于0，parents数组表示各节点的父节点
     * date: 2023/10/31 23:08
     *
     * @param parents
     * @param nums
     * @author: Chenxi Fang
     */
    public int[] smallestMissingValueSubtree(int[] parents, int[] nums) {
        int n = parents.length;
        List<Integer>[] map = new List[n];
        for (int i = 0; i < n; i++) {
            map[i] = new ArrayList<>();
        }
        //注意这里不要遍历根节点
        for (int i = 1; i < n; i++) {
            map[parents[i]].add(i);
        }
        int[] res = new int[n];
        Arrays.fill(res, 1);
        Set<Integer>[] nodes = new Set[n];
        for (int i = 0; i < n; i++) {
            nodes[i] = new HashSet<>();
        }
        dfs(0, res, map, nums, nodes);
        return res;
    }

    public int dfs(int root, int[] res, List<Integer>[] map, int[] nums, Set<Integer>[] nodes) {
        nodes[root].add(nums[root]);
        for (int child : map[root]) {
            // 先确定最小元素的初始值，为后面的穷举法剪枝
            res[root] = Math.max(res[root], dfs(child, res, map, nums, nodes));
            // 合并集合时将小集合合并到大集合
            if (nodes[root].size() < nodes[child].size()) {
                Set<Integer> temp = nodes[root];
                nodes[root] = nodes[child];
                nodes[child] = temp;
            }
            nodes[root].addAll(nodes[child]);
        }
        // 直接使用穷举法来列举最小元素
        while (nodes[root].contains(res[root])) {
            res[root]++;
        }
        return res[root];
    }

    /**
     * Description: 在限制临近条件下能围成一圈的最大人数
     * 经分析本地存在环，并为基环无向树，即各环还有延伸的分支，环大于和等于2的有不同的性质
     * date: 2023/11/1 23:36
     *
     * @param favorite
     * @author: Chenxi Fang
     */

    public int maximumInvitations(int[] favorite) {
        int n = favorite.length;
        int[] inDegrees = new int[n];
        for (int i = 0; i < n; i++) {
            inDegrees[favorite[i]]++;
        }
        boolean[] used = new boolean[n];
        int[] f = new int[n];
        Arrays.fill(f, 1);
        Deque<Integer> queue = new ArrayDeque<>();
        // 统计入度
        for (int i = 0; i < n; i++) {
            if (inDegrees[i] == 0) queue.offer(i);
        }
        // 拓扑排序加动规
        while (!queue.isEmpty()) {
            int u = queue.poll();
            used[u] = true;
            int v = favorite[u];
            f[v] = Math.max(f[v], f[u] + 1);
            inDegrees[v]--;
            if (inDegrees[v] == 0) queue.offer(v);
        }

        // 分类讨论：对于环大小等于2和大于2的情况
        int ring = 0, total = 0;
        for (int i = 0; i < n; i++) {
            if (!used[i]) {
                int j = favorite[i];
                if (favorite[j] == i) {
                    total += f[i] + f[j];
                    used[i] = used[j] = true;
                } else {
                    int u = i, cnt = 0;
                    while (true) {
                        cnt++;
                        u = favorite[u];
                        used[u] = true;
                        if (u == i) break;
                    }
                    ring = Math.max(ring, cnt);
                }
            }
        }
        return Math.max(ring, total);
    }


    @Test
    public void test() {
        System.out.println(countPoints("B0B6G0R6R0R6G9"));
    }

    public int countPoints(String rings) {
        boolean[] r = new boolean[10];
        boolean[] g = new boolean[10];
        boolean[] b = new boolean[10];
        for (int i = 0; i < rings.length(); i += 2) {
            char type = rings.charAt(i);
            int index = Integer.parseInt(String.valueOf(rings.charAt(i + 1)));
            switch (type) {
                case 'R':
                    r[index] = true;
                    break;
                case 'G':
                    g[index] = true;
                    break;
                case 'B':
                    b[index] = true;
            }
        }
        int res = 0;
        for (int i = 0; i < 10; i++) {
            if (r[i] && g[i] && b[i]) res++;
        }
        return res;
    }

    public Node connect(Node root) {
        if (root == null) return null;
        Deque<Node> queue = new ArrayDeque<>();
        queue.offer(root);
        while (!queue.isEmpty()) {
            int size = queue.size();
            for (int i = 0; i < size; i++) {
                Node cur = queue.poll();
                if (size != 1 && i != size - 1) cur.next = queue.peek();
                if (cur.left != null) queue.offer(cur.left);
                if (cur.right != null) queue.offer(cur.right);
            }
        }
        return root;
    }

    /**
     * Description: 求数组中两个数异或的最大值
     * 反向思考，由x = a ^ b 改成求 a = x ^ b，由于不超过int范围且为非负数，故最大31位数，从最高位开始，将x的位设为1，并用set存储数组对应位的数字
     * 若set中存在，说明此位x可为1，不然则为0，并依次遍历至第0位，得到最大的x值
     * 学习反向思维
     * date: 2023/11/4 19:08
     *
     * @param nums
     * @author: Chenxi Fang
     */

    public int findMaximumXOR(int[] nums) {
        int maxBit = 30;
        int x = 0;
        for (int i = maxBit; i >= 0; i--) {
            Set<Integer> set = new HashSet<>();
            for (int num : nums) {
                set.add(num >> i);
            }
            int next = 2 * x + 1;
            boolean found = false;
            for (int num : set) {
                if (set.contains(num ^ next)) {
                    found = true;
                    break;
                }
            }
            if (found) x = next;
            else x = next - 1;
        }
        return x;
    }

    @Test
    public void test1() {
        System.out.println(findChampion(4, new int[][]{{0, 2}, {1, 3}, {1, 2}}));
    }

    public int findChampion(int[][] grid) {
        int n = grid.length;
        List<Integer> list = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            list.add(i);
        }
        list.sort(((o1, o2) -> grid[o1][o2] * 2 - 1));
        return list.get(n - 1);
    }

    boolean error = false;

    public int findChampion(int n, int[][] edges) {
        int[] visited = new int[n];
        List<Integer> sorted = new ArrayList<>();
        List<Integer>[] wins = new List[n];
        for (int i = 0; i < edges.length; i++) {
            int node = edges[i][0];
            if (wins[node] == null) wins[node] = new ArrayList<>();
            wins[node].add(edges[i][1]);
        }
        for (int i = 0; i < n; i++) {
            if (visited[i] == 0) {
                dfs(i, wins, sorted, visited);
            }
        }
        System.out.println(sorted);
        System.out.println(error);
        return error ? -1 : sorted.get(n - 1);
    }

    public void dfs(int x, List<Integer>[] wins, List<Integer> sorted, int[] visited) {
        if (visited[x] == 2) {
            return;
        }
        if (visited[x] == 1) {
            error = true;
            return;
        }
        visited[x] = 1;
        if (wins[x] != null) {
            for (int next : wins[x]) {
                if (visited[next] == 0) dfs(next, wins, sorted, visited);
            }
        }
        visited[x] = 2;
        sorted.add(x);
    }

    public List<String> findRepeatedDnaSequences(String s) {
        List<String> res = new ArrayList<>();
        Map<String, Integer> map = new HashMap<>();
        for (int i = 0; i < s.length() - 9; i++) {
            String cur = s.substring(i, i + 10);
            map.put(cur, map.getOrDefault(cur, 0) + 1);
            if (map.get(cur) == 2) res.add(cur);
        }
        return res;
    }


    /**
     * Description: 求两个字符串的最大乘积，并且他们俩不含有相同的字母，字符串由小写字母组成
     * 由有限元素衍生而来的位运算，注意看题目是否元素是有限的
     * date: 2023/11/6 22:37
     *
     * @param words
     * @author: Chenxi Fang
     */

    public int maxProduct(String[] words) {
        int n = words.length;
        int[] bits = new int[n];
        // 这种有限元素组成的字符串都可以用位来表示，状态压缩，可快速判断两者是否含相同元素
        for (int i = 0; i < n; i++) {
            String word = words[i];
            for (int j = 0; j < word.length(); j++) {
                bits[i] |= 1 << word.charAt(j) - 'a';
            }
        }
        int res = 0;
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                if ((bits[i] & bits[j]) == 0) {
                    res = Math.max(res, words[i].length() * words[j].length());
                }
            }
        }
        return res;
    }

    static final int INF = 1000000000;
    int[][] dirs = new int[][]{{1, 0}, {-1, 0}, {0, 1}, {0, -1}};

    public int maximumMinutes(int[][] grid) {
        int m = grid.length, n = grid[0].length;
        int left = 0, right = m * n;
        int[][] fireTime = new int[m][n];
        for (int i = 0; i < m; i++) {
            Arrays.fill(fireTime[i], INF);
        }
        bfs(grid, fireTime);
        int ans = -1;
        while (left <= right) {
            int mid = left + (right - left) / 2;
            if (check(grid, fireTime, mid)) {
                ans = mid;
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
        return ans >= m * n ? INF : ans;
    }

    public void bfs(int[][] grid, int[][] fireTime) {
        int m = grid.length, n = grid[0].length;
        Deque<int[]> queue = new ArrayDeque<>();
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (grid[i][j] == 1) {
                    fireTime[i][j] = 0;
                    queue.offer(new int[]{i, j});
                }
            }
        }
        int time = 0;
        while (!queue.isEmpty()) {
            time++;
            int size = queue.size();
            for (int i = 0; i < size; i++) {
                int[] dim = queue.poll();
                for (int[] dir : dirs) {
                    int nx = dim[0] + dir[0];
                    int ny = dim[1] + dir[1];
                    if (nx < 0 || nx >= m || ny < 0 || ny >= n || grid[nx][ny] == 2 || fireTime[nx][ny] != INF)
                        continue;
                    fireTime[nx][ny] = time;
                    queue.offer(new int[]{nx, ny});
                }
            }
        }
    }

    public boolean check(int[][] grid, int[][] fireTime, int stayTime) {
        int m = grid.length, n = grid[0].length;
        Deque<int[]> queue = new ArrayDeque<>();
        boolean[][] visited = new boolean[m][n];
        queue.offer(new int[]{0, 0, stayTime});
        visited[0][0] = true;
        while (!queue.isEmpty()) {
            int[] dim = queue.poll();
            int x = dim[0], y = dim[1], time = dim[2];
            for (int[] dir : dirs) {
                int nx = x + dir[0];
                int ny = y + dir[1];
                if (nx < 0 || nx >= m || ny < 0 || ny >= n || grid[nx][ny] == 2 || visited[nx][ny]) continue;
                if (nx == m - 1 && ny == n - 1) return fireTime[nx][ny] >= time + 1;
                if (fireTime[nx][ny] > time + 1) {
                    queue.offer(new int[]{nx, ny, time + 1});
                    visited[nx][ny] = true;
                }
            }
        }
        return false;
    }

    public int[] successfulPairs(int[] spells, int[] potions, long success) {
        int n = spells.length;
        Arrays.sort(potions);
        int[] res = new int[n];
        for (int i = 0; i < n; i++) {
            res[i] = successfulPair(spells[i], potions, success);
        }
        return res;
    }

    public int successfulPair(int spell, int[] potions, long success) {
        int left = 0, right = potions.length - 1;
        while (left <= right) {
            int mid = left + (right - left) / 2;
            if ((long) spell * potions[mid] >= success) {
                right = mid - 1;
            } else {
                left = mid + 1;
            }
        }
        return potions.length - left;
    }

    /**
     * Description: 结果的连通分量必是N/2, 连通分量相减即可
     * leetcode 765
     * date: 2023/11/12 19:43
     *
     * @param row
     * @author: Chenxi Fang
     */
    public int minExchange(int[] row) {
        int len = row.length;
        int N = len / 2;
        UnionFind unionFind = new UnionFind(N);
        for (int i = 0; i < len; i += 2) {
            unionFind.merge(row[i] / 2, row[i + 1] / 2);
        }
        return N - unionFind.getCount();
    }

    @Test
    public void test11() {
        System.out.println(findCity(4, new int[][]{{0, 1, 3}, {1, 2, 1}, {1, 3, 4}, {2, 3, 1}}, 4));
    }

    public int findCity(int n, int[][] edges, int distanceThreshold) {
        Map<Integer, List<int[]>> map = new HashMap<>();
        for (int[] edge : edges) {
            if (!map.containsKey(edge[0])) map.put(edge[0], new ArrayList<>());
            if (!map.containsKey(edge[1])) map.put(edge[1], new ArrayList<>());
            map.get(edge[0]).add(new int[]{edge[1], edge[2]});
            map.get(edge[1]).add(new int[]{edge[0], edge[2]});
        }
        int minNeighbors = Integer.MAX_VALUE;
        int res = -1;
        for (int i = 0; i < n; i++) {
            if (map.get(i) == null) continue;
            int neighbors = 0;
            boolean[] visited = new boolean[n];
            visited[i] = true;
            for (int[] path : map.get(i)) {
                neighbors += dfs(path[0], distanceThreshold - path[1], map, visited);
            }
            if (minNeighbors >= neighbors) {
                minNeighbors = neighbors;
                res = i;
            }
        }
        return res;

    }

    public int dfs(int index, int threshold, Map<Integer, List<int[]>> map, boolean[] visited) {
        if (threshold < 0 || visited[index] || !map.containsKey(index)) return 0;
        visited[index] = true;
        int sum = 0;
        for (int[] path : map.get(index)) {
            sum += dfs(path[0], threshold - path[1], map, visited);
        }
        return 1 + sum;
    }

    int k;

    public int[] maxSumOfThreeSubarray(int[] nums, int k) {
        this.k = k;
        int n = nums.length;
        int[] ans = new int[3];
        if (n < 3) return ans;
        int[] preSum = new int[n];
        int sum = 0;
        for (int i = 0; i < n; i++) {
            sum += nums[i];
            preSum[i] = sum;
        }
        backtracking(nums, 0);
        int max = Integer.MIN_VALUE;
        for (List<Integer> path : res) {
            int threeSum = 0;
            for (int start : path) {
                if (start == 0) {
                    threeSum += preSum[start + k - 1];
                } else {
                    threeSum += preSum[start + k - 1] - preSum[start - 1];
                }
            }
            if (threeSum > max) {
                max = threeSum;
                for (int i = 0; i < 3; i++) {
                    ans[i] = path.get(i);
                }
            }
        }
        return ans;
    }

    public void backtracking(int[] nums, int start) {
        if (path.size() == 3) {
            res.add(new ArrayList<>(path));
            return;
        }
        for (int i = start; i < nums.length - k * (3 - path.size()); i++) {
            path.add(i);
            backtracking(nums, i + k);
            path.removeLast();
        }
    }

    public int minDeletion(int[] nums) {
        int n = nums.length;
        int left = 0, right = 0;
        int res = 0;
        while (left < n) {
            right = left + 1;
            while (right < n && nums[right] == nums[left]) right++;
            res += right - left - 1;
            left = right + 1;
        }
        return right >= n ? res + 1 : res;
    }

    public int minPathCost(int[][] grid, int[][] moveCost) {
        int m = grid.length, n = grid[0].length;
        int[][] dp = new int[m][n];
        for (int[] arr : dp) {
            Arrays.fill(arr, Integer.MAX_VALUE);
        }
        for (int i = 0; i < n; i++) {
            dp[0][i] = grid[0][i];
        }
        for (int i = 1; i < m; i++) {
            for (int j = 0; j < n; j++) {
                for (int k = 0; k < n; k++) {
                    dp[i][j] = Math.min(dp[i][j], dp[i - 1][k] + moveCost[grid[i - 1][k]][j] + grid[i][j]);
                }
            }
        }
        int res = Integer.MAX_VALUE;
        for (int i = 0; i < n; i++) {
            res = Math.min(res, dp[m - 1][i]);
        }
        return res;
    }

    String[] keys = new String[]{"quot", "apos", "amp", "gt", "lt", "frasl"};
    String[] values = new String[]{"\"", "'", "&", ">", "<", "/"};

    public String entityParser(String text) {
        int n = text.length();
        int left = 0, right = 0;
        StringBuilder res = new StringBuilder();
        Map<String, String> map = new HashMap<>();
        for (int i = 0; i < keys.length; i++) {
            map.put(keys[i], values[i]);
        }
        while (left < n) {
            char c = text.charAt(left);
            if (c == '&') {
                right = left + 1;
                while (right < n && isCharacter(text, right)) {
                    right++;
                }
                if (right < n && text.charAt(right) == ';' && map.containsKey(text.substring(left + 1, right))) {
                    res.append(map.get(text.substring(left + 1, right)));
                    left = right + 1;
                    continue;
                }
            }
            res.append(c);
            left++;
        }
        return res.toString();
    }

    boolean isCharacter(String s, int index) {
        char c = s.charAt(index);
        return c >= 'a' && c <= 'z';
    }

    int ans;
    LinkedList<Integer> nodePath = new LinkedList<>();

    public int pseudoPalindromicPaths(TreeNode root) {
        if (root == null) return 0;
        backtracking(root);
        return ans;
    }


    public void backtracking(TreeNode root) {
        nodePath.add(root.val);
        if (root.left == null && root.right == null) {
            if (isValid(nodePath)) ans++;
        }
        if (root.left != null) backtracking(root.left);
        if (root.right != null) backtracking(root.right);
        nodePath.removeLast();
    }

    public boolean isValid(List<Integer> path) {
        int size = path.size();
        int[] count = new int[10];
        for (int i = 0; i < size; i++) {
            count[path.get(i)]++;
        }
        int single = 0;
        for (int i = 0; i < 10; i++) {
            if (count[i] % 2 != 0) {
                single++;
                if (single >= 2) return false;
            }
        }
        System.out.println("--");
        return true;
    }

    public int uniqueLetterString(String s) {
        int[][] set = new int[26][2];
        for (int i = 0; i < 26; i++) {
            Arrays.fill(set[i], -1);
        }
        int n = s.length();
        int[] dp = new int[n];
        dp[0] = 1;
        set[s.charAt(0) - 'A'][0] = 0;
        int res = 1;
        char prev = s.charAt(0);
        for (int i = 1; i < n; i++) {
            char c = s.charAt(i);
            int lastIndex = set[c - 'A'][0];
            int lastLastIndex = set[c - 'A'][1];
            if (lastIndex == -1) {
                dp[i] = dp[i - 1] + i + 1;
            } else {
                dp[i] = dp[i - 1] + (i - lastIndex - 1) + 1 - (lastIndex - lastLastIndex);
            }
            res += dp[i];
            set[c - 'A'][0] = i;
            set[c - 'A'][1] = lastIndex;
        }
        return res;
    }

    @Test
    public void test111() {
        System.out.println(sumSubarrayMin(new int[]{11, 81, 94, 43, 3}));
    }

    public int sumSubarrayMin(int[] arr) {
        int n = arr.length;
        int[] dp = new int[n];
        long ans = 0;
        Deque<Integer> stack = new ArrayDeque<>();
        final int MOD = 1000000007;
        for (int i = 0; i < n; i++) {
            while (!stack.isEmpty() && arr[stack.peek()] > arr[i]) {
                stack.pop();
            }
            int k = stack.isEmpty() ? (i + 1) : (i - stack.peek());
            dp[i] = k * arr[i] + (stack.isEmpty() ? 0 : dp[i - k]);
            ans = (ans + dp[i]) % MOD;
            stack.push(i);
        }
        return (int) ans;
    }


}

class TreeNode {

    int val;
    TreeNode left;
    TreeNode right;

    public TreeNode(int val) {
        this.val = val;
    }
}

class Automation {
    public long ans = 0;
    public int sign = 1;
    private String state = "start";
    private Map<String, String[]> table = new HashMap<>();


    public Automation() {
        table.put("start", new String[]{"start", "signed", "in_number", "end"});
        table.put("signed", new String[]{"end", "end", "in_number", "end"});
        table.put("in_number", new String[]{"end", "end", "in_number", "end"});
        table.put("end", new String[]{"end", "end", "end", "end"});
    }

    public void get(char c) {
        state = table.get(state)[getCol(c)];
        if ("in_number".equals(state)) {
            ans = ans * 10 + c - '0';
            ans = sign == 1 ? Math.min(ans, Integer.MAX_VALUE) : Math.min(ans, -(long) Integer.MIN_VALUE);
        } else if ("signed".equals(state)) {
            sign = c == '-' ? -1 : 1;
        }
    }

    public int getCol(char c) {
        if (c == ' ') return 0;
        else if (c == '+' || c == '-') return 1;
        else if (c >= '0' && c <= '9') return 2;
        else return 3;
    }

    public boolean closeStrings(String word1, String word2) {
        int n = word1.length();
        if (n != word2.length()) return false;
        int[] chars1 = new int[26];
        int[] chars2 = new int[26];
        Map<Integer, Integer> map = new HashMap<>();
        for (int i = 0; i < n; i++) {
            chars1[word1.charAt(i) - 'a']++;
        }
        for (int i = 0; i < 26; i++) {
            if (chars1[i] != 0) map.put(chars1[i], map.getOrDefault(chars1[i], 0) + 1);
        }
        for (int i = 0; i < n; i++) {
            char c = word2.charAt(i);
            if (chars1[c - 'a'] == 0) return false;
            chars2[c - 'a']++;
        }
        for (int i = 0; i < 26; i++) {
            if (chars2[i] == 0) continue;
            if (!map.containsKey(chars2[i]) || map.get(chars2[i]) == 0) return false;
            map.put(chars2[i], map.get(chars2[i]) - 1);
        }
        for (int value : map.values()) {
            if (value != 0) return false;
        }
        return true;
    }

    int val = 0;

    public TreeNode bstToGst(TreeNode root) {
        dfs(root);
        return root;
    }

    public void dfs(TreeNode root) {
        if (root == null) return;
        dfs(root.right);
        root.val += val;
        val = root.val;
        dfs(root.left);
    }

    public int maxScore(int[] cardPoints, int k) {
        int n = cardPoints.length;
        int min = Integer.MAX_VALUE;
        int sum = 0;
        int winSum = 0;
        for (int i = 0; i < n; i++) {
            winSum += cardPoints[i];
            sum += cardPoints[i];
            if (i == n - k - 1) min = sum;
            if (i >= n - k) {
                sum -= cardPoints[i - (n - k)];
                min = Math.min(min, sum);
            }
        }
        return winSum - min;
    }

    public boolean carPooling(int[][] trips, int capacity) {
        Arrays.sort(trips, ((o1, o2) -> o2[1] - o1[1]));
        int[] dis = new int[1001];
        for (int i = 0; i < trips.length; i++) {
            for (int j = trips[i][1]; j < trips[i][2]; j++) {
                dis[j] += trips[i][0];
                if (dis[j] > capacity) return false;
            }
        }
        return true;
    }

    public int firstCompleteIndex(int[] arr, int[][] mat) {
        int[] map = new int[100001];
        for (int i = 0; i < arr.length; i++) {
            map[arr[i]] = i;
        }
        int m = mat.length, n = mat[0].length;
        int minIndex = Integer.MAX_VALUE;
        for (int i = 0; i < m; i++) {
            int maxIndex = -1;
            for (int j = 0; j < n; j++) {
                maxIndex = Math.max(map[mat[i][j]], maxIndex);
            }
            minIndex = Math.min(minIndex, maxIndex);
        }
        for (int i = 0; i < n; i++) {
            int maxIndex = -1;
            for (int j = 0; j < m; j++) {
                maxIndex = Math.max(map[mat[j][i]], maxIndex);
            }
            minIndex = Math.min(minIndex, maxIndex);
        }
        return minIndex;
    }


    long res = 0;

    public long minimumFuelCost(int[][] roads, int seats) {
        int n = roads.length + 1;
        List<Integer>[] map = new List[n];
        for (int i = 0; i < n; i++) {
            map[i] = new ArrayList<>();
        }
        for (int[] road : roads) {
            map[road[0]].add(road[1]);
            map[road[1]].add(road[0]);
        }
        dfs(0, -1, map, seats);
        return res;
    }

    public int dfs(int cur, int dest, List<Integer>[] map, int seats) {
        int peopleSum = 1;
        for (int next : map[cur]) {
            // 防止将目的节点当作子节点，因为map是双向的
            if (next != dest) {
                int peopleCnt = dfs(next, cur, map, seats);
                peopleSum += peopleCnt;
                // 向上取整
                res += (peopleCnt + seats - 1) / seats;
            }
        }
        return peopleSum;
    }

    public int minimumTotalPrice(int n, int[][] edges, int[] price, int[][] trips) {
        List<Integer>[] map = new List[n];
        for (int i = 0; i < n; i++) {
            map[i] = new ArrayList<>();
        }
        for (int[] edge : edges) {
            map[edge[0]].add(edge[1]);
            map[edge[1]].add(edge[0]);
        }
        // 在trips中经过的每个节点经过的次数，次数乘以节点的值即为价格
        int[] count = new int[n];
        for (int[] trip : trips) {
            dfs(map, count, trip[0], -1, trip[1]);
        }
        int[] pair = dp(map, count, price, 0, -1);
        return Math.min(pair[0], pair[1]);
    }

    public boolean dfs(List<Integer>[] map, int[] count, int start, int parent, int end) {
        if (start == end) {
            count[start]++;
            return true;
        }
        // 在进行深搜时，只要是包含end的子树的根节点，count都加一
        for (int child : map[start]) {
            if (child == parent) continue;
            if (dfs(map, count, child, start, end)) {
                count[start]++;
                return true;
            }
        }
        return false;
    }

    public int[] dp(List<Integer>[] map, int[] count, int[] price, int start, int parent) {
        // 保持原价格和价格减半的状态，每个节点都只能是这两个状态
        int[] res = new int[]{count[start] * price[start], count[start] * price[start] / 2};
        for (int child : map[start]) {
            if (child == parent) continue;
            // 0和1分别对应两种状态
            int[] pair = dp(map, count, price, child, start);
            res[0] += Math.min(pair[0], pair[1]);
            res[1] += pair[0];
        }
        return res;
    }

    public int minReorder(int n, int[][] connections) {
        List<int[]>[] map = new List[n];
        for (int i = 0; i < n; i++) {
            map[i] = new ArrayList<>();
        }
        for (int[] connection : connections) {
            map[connection[0]].add(new int[]{connection[1], 1});
            map[connection[1]].add(new int[]{connection[0], 0});
        }
        return dfs(0, -1, map);
    }

    public int dfs(int cur, int parent, List<int[]>[] map){
        int res = 0;
        for (int[] next : map[cur]) {
            if(next[0] == parent) continue;
            res += next[1] + dfs(next[0], cur, map);
        }
        String s = "hello";
        StringBuilder s1 = new StringBuilder(s);
        return res;
    }


}

class SmallestInfiniteSet {
    int lastNum;
    PriorityQueue<Integer> pq;
    Set<Integer> set;

    public SmallestInfiniteSet() {
        lastNum = 1;
        pq = new PriorityQueue<Integer>();
        set = new HashSet<>();
    }

    public int popSmallest() {
        if (pq.isEmpty()) return lastNum++;
        int num = pq.poll();
        set.remove(num);
        return num;
    }

    public void addBack(int num) {
        if (num >= lastNum) return;
        if (!set.contains(num)) {
            pq.add(num);
            set.add(num);
        }
    }
}

class Node {
    int val;
    Node left;
    Node right;
    Node next;
}

class UnionFind {
    private int[] parent;
    private int[] root;
    private int count;

    public int getCount() {
        return count;
    }

    public UnionFind(int n) {
        count = n;
        parent = new int[n];
        root = new int[n];
        for (int i = 0; i < n; i++) {
            parent[i] = i;
        }
    }

    public int getParent(int x) {
        if (x == parent[x]) return x;
        return parent[x] = getParent(parent[x]);
    }

    public void merge(int x, int y) {
        if (parent[x] == parent[y]) return;
        count--;
        if (root[x] > root[y]) parent[y] = x;
        else {
            parent[x] = y;
            root[y]++;
        }
    }
}

class NumArray {
    public int[] tree;
    public int n;
    public int[] nums;

    public int lastSub(int x) {
        return x & (-x);
    }

    public NumArray(int[] nums) {
        this.n = nums.length;
        tree = new int[n + 1];
        for (int i = 0; i < n; i++) {
            update1(i, nums[i]);
        }
        this.nums = nums;
    }

    public void update1(int index, int val) {
        for (int i = index + 1; i <= n; i += lastSub(i)) {
            tree[i] += val;
        }
    }

    public void update(int index, int val) {
        int gap = val - nums[index];
        nums[index] = val;
        if (gap == 0) return;
        update1(index, gap);
    }

    public int sumRange(int left, int right) {
        if (left == 0) return sum(right + 1);
        return sum(right + 1) - sum(left);
    }

    public int sum(int k) {
        int ans = 0;
        for (int i = k; i > 0; i -= lastSub(i)) {
            ans += tree[i];
        }
        System.out.println(ans);
        return ans;
    }
}

class CountIntervals {
    private int cnt;
    private TreeMap<Integer, Integer> map = new TreeMap<>();

    public CountIntervals() {
    }

    public void add(int left, int right) {
        Map.Entry<Integer, Integer> interval = map.floorEntry(right);
        while (interval != null && interval.getValue() >= left){
            int l = interval.getKey(), r = interval.getValue();
            left = Math.min(left, l);
            right = Math.max(right, r);
            cnt -= r - l + 1;
            map.remove(l);
            interval = map.floorEntry(right);
        }
        cnt += right - left + 1;
        map.put(left, right);
    }

    public int count() {
        return cnt;
    }


}

