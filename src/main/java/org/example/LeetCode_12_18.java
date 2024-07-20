package org.example;

import org.junit.Test;
import sun.rmi.runtime.Log;

import javax.swing.plaf.metal.MetalTheme;
import java.util.*;
import java.util.concurrent.ForkJoinPool;
import java.util.stream.Collectors;
import java.util.stream.Stream;

/**
 * @program: Algorithm
 * @description: 每日一题
 * @author: Chenxi Fang
 * @create: 2023-12-18
 **/
public class LeetCode_12_18 {
    int n;

    public int findPeakElement(int[] nums) {
        n = nums.length;
        if (n == 1) return nums[0];
        return search(nums, 0, n - 1);
    }

    public int search(int[] nums, int left, int right) {
        if (left > right) return -1;
        int mid = left + (right - left) / 2;
        if ((mid == 0 && nums[mid] > nums[mid + 1]) || (mid == n - 1 && nums[mid] > nums[mid - 1]) ||
                (mid != 0 && mid != n - 1 && nums[mid] > nums[mid - 1] && nums[mid] > nums[mid + 1])) {
            return mid;
        }
        int leftPeak = search(nums, left, mid - 1);
        return leftPeak != -1 ? leftPeak : search(nums, mid + 1, right);
    }


    public int[] findPeakGrid(int[][] mat) {
        m = mat.length;
        n = mat[0].length;
        if (m == 1 && n == 1) return new int[]{0, 0};
        int[][] borderedMat = new int[m + 2][n + 2];
        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                borderedMat[i][j] = mat[i - 1][j - 1];
            }
        }
        int[] res = null;
        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                res = searchMesh(borderedMat, i, 1, n);
                if (res != null) return res;
            }
        }
        return res;

    }

    int m;

    public int[] searchMesh(int[][] mat, int row, int left, int right) {
        if (left > right) return null;
        int mid = left + (right - left) / 2;
        int num = mat[row][mid];
        if (num > mat[row - 1][mid] && num > mat[row + 1][mid] && num > mat[row][mid - 1] && num > mat[row][mid + 1]) {
            return new int[]{row, mid};
        }
        int[] leftPeak = searchMesh(mat, row, left, mid - 1);
        return leftPeak != null ? leftPeak : searchMesh(mat, row, mid + 1, right);
    }

    public long maximumSumOfHeights(List<Integer> maxHeights) {
        int n = maxHeights.size();
        long res = 0;
        Deque<Integer> stack1 = new ArrayDeque<>();
        Deque<Integer> stack2 = new ArrayDeque<>();
        long[] prefix = new long[n];
        long[] suffix = new long[n];
        for (int i = 0; i < n; i++) {
            while (!stack1.isEmpty() && maxHeights.get(i) < maxHeights.get(stack1.peek())) {
                stack1.pop();
            }
            if (stack1.isEmpty()) prefix[i] = (long) (i + 1) * maxHeights.get(i);
            else {
                prefix[i] = prefix[stack1.peek()] + (long) (i - stack1.peek()) * maxHeights.get(i);
            }
            stack1.push(i);
        }
        for (int i = n - 1; i >= 0; i--) {
            while (!stack2.isEmpty() && maxHeights.get(i) < maxHeights.get(stack2.peek())) {
                stack2.pop();
            }
            if (stack2.isEmpty()) suffix[i] = (long) (n - i) * (maxHeights.get(i));
            else {
                suffix[i] = suffix[stack2.peek()] + (long) (stack2.peek() - i) * maxHeights.get(i);
            }
            stack2.push(i);
            res = Math.max(prefix[i] + suffix[i] - maxHeights.get(i), res);
        }
        return res;
    }

    @Test
    public void test() {
        System.out.println(minStoneSum(new int[]{10000}, 10000));
    }

    public int minStoneSum(int[] piles, int k) {
        int sum = Arrays.stream(piles).sum();
        int n = piles.length;
        PriorityQueue<Integer> queue1 = new PriorityQueue<>();
        int size = Math.min(k, n);
        for (int pile : piles) {
            queue1.offer(pile);
            if (queue1.size() > size) queue1.poll();
        }
        PriorityQueue<Integer> queue2 = new PriorityQueue<>(((o1, o2) -> o2 - o1));
        while (!queue1.isEmpty()) queue2.offer(queue1.poll());
        int subtract = 0;
        for (int i = 0; i < k; i++) {
            int max = queue2.poll();
            int halfMax = max / 2;
            subtract += halfMax;
            queue2.offer(max - halfMax);
        }
        return sum - subtract;
    }

    public List<Integer> numOfBurgers(int tomatoSlices, int cheeseSlices) {
        ArrayList<Integer> res = new ArrayList<>();
        int x = tomatoSlices - cheeseSlices * 2;
        if (x < 0 || x % 2 != 0) return res;
        int y = tomatoSlices - x * 2;
        if (y < 0 || y % 2 != 0) return res;
        res.add(x / 2);
        res.add(y / 2);
        return res;
    }

    public int maxStudents(char[][] seats) {
        int m = seats.length;
        int n = seats[0].length;
        int[][] dp = new int[m + 1][n + 1];
        if (seats[0][0] == '.') dp[1][1] = 1;
        for (int i = 1; i < m; i++) {
            if (seats[i][0] == '#') dp[i + 1][1] = dp[i][1];
            else dp[i + 1][1] = Math.max(dp[i][1], dp[i - 1][1] + 1);
        }
        for (int i = 1; i < n; i++) {
            if (seats[0][i] == '#') dp[1][i + 1] = dp[1][i];
            else dp[1][i + 1] = Math.max(dp[1][i], dp[1][i - 1] + 1);
        }
        for (int i = 1; i < m; i++) {
            for (int j = 1; j < n; j++) {
                if (seats[i][j] == '#') dp[i + 1][j + 1] = Math.max(dp[i][j + 1], Math.max(dp[i][j], dp[i + 1][j]));
                else dp[i + 1][j + 1] = Math.max(dp[i][j + 1] + 1, Math.max(dp[i + 1][j], dp[i][j]));
            }
        }
        return dp[m][n];
    }

    public long minCost(int[] nums, int x) {
        int n = nums.length;
        long[] s = new long[n];
        for (int i = 0; i < n; i++) {
            s[i] = (long) x * i;
        }
        for (int i = 0; i < n; i++) {
            int min = nums[i];
            for (int j = i; j < n + i; j++) {
                min = Math.min(min, nums[j % n]);
                s[j - i] += min;
            }
        }
        long res = Long.MAX_VALUE;
        for (int i = 0; i < n; i++) {
            res = Math.min(res, s[i]);
        }
        return res;
    }

    public int minOperationsMaxProfit(int[] customers, int boardingCost, int runningCost) {
        int n = customers.length;
        int[] cusList = new int[n];
        if (boardingCost * 4 < runningCost) return -1;
        int prev = 0;
        for (int i = 0; i < n; i++) {
            int cur = customers[i] + prev;
            if (cur > 4) {
                cusList[i] = 4;
                prev = cur - 4;
            } else {
                cusList[i] = cur;
                prev = 0;
            }
        }
        int maxProfit = 0, profit = 0, maxOpe = 0;
        for (int i = 0; i < n; i++) {
            profit += cusList[i] * boardingCost - runningCost;
            if (maxProfit < profit) {
                maxProfit = profit;
                maxOpe = i + 1;
            }
        }
        if (prev != 0) {
            for (int i = 0; i < prev / 4; i++) {
                profit += 4 * boardingCost - runningCost;
                if (maxProfit < profit) {
                    maxProfit = profit;
                    maxOpe = i + n + 1;
                }
            }
            if (maxProfit < profit + (prev % 4) * boardingCost - runningCost) {
                maxProfit = profit;
                maxOpe++;
            }
        }
        return maxProfit == 0 ? -1 : maxOpe;
    }

    public int getMaxRepetitions(String s1, int n1, String s2, int n2) {
        int l1 = s1.length(), l2 = s2.length();
        int cnt = 1;
        s1 = stringRepeat(s1, n1);
        s2 = stringRepeat(s2, n2);
        for (int i = (l1 * n1) / (l2 * n2); i >= 1; i--) {
            int len = maxSubList(s1, stringRepeat(s2, i));
            if (len == s2.length() * i) {
                cnt = i;
                break;
            }
        }
        return cnt;
    }

    public int maxSubList(String s1, String s2) {
        int n1 = s1.length(), n2 = s2.length();
        int[][] dp = new int[n1 + 1][n2 + 1];
        for (int i = 1; i <= n1; i++) {
            for (int j = 1; j <= n2; j++) {
                if (s1.charAt(i - 1) == s2.charAt(j - 1)) {
                    dp[i][j] = dp[i - 1][j - 1] + 1;
                } else {
                    dp[i][j] = Math.max(dp[i - 1][j], dp[i][j - 1]);
                }
            }
        }
        return dp[n1][n2];
    }

    public String stringRepeat(String s, int count) {
        StringBuilder res = new StringBuilder();
        for (int i = 0; i < count; i++) {
            res.append(s);
        }
        return res.toString();
    }

    public ListNode removeNodes(ListNode head) {
        ListNode dummy = new ListNode(0);
        dummy.next = head;
        Deque<ListNode> stack = new ArrayDeque<>();
        while (head != null) {
            while (!stack.isEmpty() && stack.peek().val < head.val) {
                stack.pop();
            }
            if (stack.isEmpty()) dummy.next = head;
            else stack.peek().next = head;
            stack.push(head);
            head = head.next;
        }
        return dummy.next;
    }


    List<List<Integer>> res = new ArrayList<>();
    LinkedList<Integer> path = new LinkedList<>();

    public int maximumRows(int[][] matrix, int numSelect) {
        int m = matrix.length, n = matrix[0].length;
        backtracking(n, numSelect, 0);
        int maxRows = 0;
        for (List<Integer> path : res) {
            maxRows = Math.max(maxRows, calRows(path, matrix, m, n));
        }
        return maxRows;
    }

    public void backtracking(int n, int numSelect, int index) {
        if (path.size() == numSelect) {
            res.add(new ArrayList<>(path));
            return;
        }
        for (int i = index; i < n - (numSelect - path.size()) + 1; i++) {
            path.add(i);
            backtracking(n, numSelect, i + 1);
            path.removeLast();
        }
    }

    public int calRows(List<Integer> path, int[][] matrix, int m, int n) {
        Set<Integer> set = new HashSet<>(path);
        int cnt = 0;
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (matrix[i][j] == 1 && !set.contains(j)) break;
                if (j == n - 1) cnt++;
            }
        }
        return cnt;
    }


    public int[] canSeePersonsCount(int[] heights) {
        int n = heights.length;
        int[] res = new int[n];
        Deque<Integer> stack = new ArrayDeque<>();
        for (int i = n - 1; i >= 0; i--) {
            while (!stack.isEmpty() && stack.peek() < heights[i]) {
                stack.pop();
                res[i]++;
            }
            if (!stack.isEmpty()) res[i]++;
            stack.push(heights[i]);
        }
        return res;
    }

    public ListNode insertGreatestCommonDivisors(ListNode head) {
        if (head == null || head.next == null) return head;
        ListNode cur = head;
        ListNode next = head.next;
        while (next != null) {
            ListNode insertNode = new ListNode(calGreatestCommonDivisors(cur.val, next.val));
            cur.next = insertNode;
            insertNode.next = next;
            cur = next;
            next = next.next;
        }
        return head;
    }

    public int calGreatestCommonDivisors(int a, int b) {
        int min = Math.min(a, b);
        for (int i = min; i > 1; i--) {
            if ((a % i == 0) && (b % i == 0)) return i;
        }
        return 1;
    }

    public int numberOfBoomerangs(int[][] points) {
        int res = 0;
        for (int[] p : points) {
            HashMap<Integer, Integer> map = new HashMap<>();
            for (int[] q : points) {
                int dis = (p[0] - q[0]) * (p[0] - q[0]) + (p[1] - q[1]) * (p[1] - q[1]);
                map.put(dis, map.getOrDefault(dis, 0) + 1);
            }
            for (Map.Entry<Integer, Integer> entry : map.entrySet()) {
                int m = entry.getValue();
                res += m * (m - 1);
            }
        }
        return res;
    }

    public int minExtraChar(String s, String[] dictionary) {
        int n = s.length();
        int[] dp = new int[n + 1];
        dp[0] = 0;
        Set<String> set = new HashSet<>(Arrays.asList(dictionary));
        for (int i = 1; i <= n; i++) {
            dp[i] = dp[i - 1] + 1;
            for (int j = 1; j <= i; j++) {
                if (set.contains(s.substring(j - 1, i))) {
                    dp[i] = Math.min(dp[i], dp[j - 1]);
                }
            }
        }
        return dp[n];
    }

    public int minLength(String s) {
        Deque<Character> stack = new ArrayDeque<>();
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            if (!stack.isEmpty() && ((c == 'B' && stack.peek() == 'A') || (c == 'D' && stack.peek() == 'C'))) {
                stack.pop();
                continue;
            }
            stack.push(c);
        }
        return stack.size();
    }

    public int addMinimum(String word) {
        int n = word.length();
        int res = 1;
        for (int i = 1; i < n; i++) {
            if (word.charAt(i) < word.charAt(i - 1)) res++;
        }
        return res * 3 - n;
    }

    public int countWords(String[] words1, String[] words2) {
        HashMap<String, Integer> map1 = new HashMap<>();
        for (String s : words1) {
            map1.put(s, map1.getOrDefault(s, 0) + 1);
        }
        HashMap<String, Integer> map2 = new HashMap<>();
        for (String s : words1) {
            map2.put(s, map2.getOrDefault(s, 0) + 1);
        }
        System.out.println(map1);
        System.out.println(map2);
        int res = 0;
        for (Map.Entry<String, Integer> entry : map1.entrySet()) {
            if (entry.getValue() == 1) {
                String s = entry.getKey();
                if (map2.containsKey(s) && map2.get(s) == 1) res++;
            }
        }
        return res;
    }

    public ListNode deleteDuplicates(ListNode head) {
        if (head == null || head.next == null) return head;
        ListNode dummy = new ListNode(0);
        dummy.next = head;
        HashSet<Integer> set = new HashSet<>();
        while (head.next != null) {
            if (head.val == head.next.val) {
                head.next = head.next.next;
                set.add(head.val);
            } else {
                head = head.next;
            }
        }
        ListNode cur = dummy;
        while (cur.next != null) {
            if (set.contains(cur.next.val)) {
                cur.next = cur.next.next;
            } else {
                cur = cur.next;
            }
        }
        return dummy.next;
    }

    public int maximumNumberOfStringPairs(String[] words) {
        Set<String> set = new HashSet<>();
        int res = 0;
        for (String word : words) {
            if (!set.contains(word)) {
                set.add(reverseStr(word));
            } else {
                set.remove(word);
                res++;
            }
        }
        return res;
    }

    public String reverseStr(String word) {
        StringBuilder res = new StringBuilder();
        for (int i = word.length() - 1; i >= 0; i--) {
            res.append(word.charAt(i));
        }
        return res.toString();
    }

    public long minimumRemoval(int[] beans) {
        Arrays.sort(beans);
        int n = beans.length;
        long sum = 0;
        long max = 0;
        for (int i = 0; i < n; i++) {
            sum += beans[i];
            max = Math.max(max, (long) beans[i] * (n - i));
        }
        return sum - max;
    }

    public int minimumTime(List<Integer> nums1, List<Integer> nums2, int x) {
        int n = nums1.size();
        int s1 = 0, s2 = 0;
        int[][] pair = new int[n][2];
        for (int i = 0; i < n; i++) {
            int a = nums1.get(i);
            int b = nums2.get(i);
            pair[i][0] = a;
            pair[i][1] = b;
            s1 += a;
            s2 += b;
        }
        Arrays.sort(pair, (o1, o2) -> o1[1] - o2[1]);
        int[] dp = new int[n + 1];
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j > 0; j--) {
                dp[j] = Math.max(dp[j], dp[j - 1] + pair[i][0] + j * pair[i][1]);
            }
        }
        for (int i = 0; i <= n; i++) {
            if (s1 + s2 * i - dp[i] <= x) {
                return i;
            }
        }
        return -1;
    }

    public List<String> splitWordsBySeparator(List<String> words, char separator) {
        List<String> res = new ArrayList<>();
        for (String word : words) {
            split(word, separator, res);
        }
        return res;
    }

    public void split(String word, char separator, List<String> res) {
        int left = 0;
        int n = word.length();
        for (int right = 0; right < n; right++) {
            if (word.charAt(right) == separator) {
                if (right > left) {
                    res.add(word.substring(left, right));
                }
                left = right + 1;
            }
        }
        if (word.charAt(n - 1) != separator) {
            res.add(word.substring(left, n));
        }
    }

    @Test
    public void splitTest() {
        List<String> strings = new ArrayList<>();
        split(".one.two.three.", '.', strings);
        System.out.println(strings);
    }

    int kSplit;
    int nSplit;
    List<List<Integer>> resSplit = new ArrayList<>();
    LinkedList<Integer> pathSplit = new LinkedList<>();

    public int splitArray(int[] nums, int k) {
        int n = nums.length;
        int[][] f = new int[n + 1][k + 1];
        for (int i = 0; i <= n; i++) {
            Arrays.fill(f[i], Integer.MAX_VALUE);
        }
        int[] sub = new int[n + 1];
        for (int i = 0; i < n; i++) {
            sub[i + 1] = sub[i] + nums[i];
        }
        f[0][0] = 0;
        for (int i = 1; i <= n; i++) {
            for (int j = 1; j <= Math.min(i, k); j++) {
                for (int l = 0; l < i; l++) {
                    f[i][j] = Math.min(f[i][j], Math.max(f[l][j - 1], sub[i] - sub[l]));
                }
            }
        }
        return f[n][k];
    }

    public void backtrackingSplitArray(int[] nums, int index) {
        if (index == nums.length && pathSplit.size() == kSplit) {
            resSplit.add(new ArrayList<>(pathSplit));
            return;
        }
        for (int i = index; i < nums.length - (kSplit - pathSplit.size()); i++) {
            if (pathSplit.size() == (kSplit - 1)) {
                pathSplit.add(calSubArrSum(nums, index, nSplit - 1));
                backtrackingSplitArray(nums, nSplit);
                pathSplit.removeLast();
                break;
            }
            pathSplit.add(calSubArrSum(nums, index, i));
            backtrackingSplitArray(nums, i + 1);
            pathSplit.removeLast();
        }
    }

    private int calSubArrSum(int[] nums, int start, int end) {
        int sum = 0;
        for (int i = start; i <= end; i++) {
            sum += nums[i];
        }
        return sum;
    }

    private int calMaxSum(List<Integer> path) {
        int max = Integer.MIN_VALUE;
        for (int i : path) {
            max = Math.max(max, i);
        }
        return max;
    }

    public int maximumSwap(int num) {
        if (num == 0) return 0;
        List<Integer> list = new ArrayList<>();
        int dummy = num;
        while (num != 0) {
            list.add(num % 10);
            num /= 10;
        }
        Collections.reverse(list);
        System.out.println(list);
        Deque<Integer> stack = new ArrayDeque<>();
        for (int i : list) {
            while (!stack.isEmpty() && stack.peek() < i) {
                stack.pop();
            }
            stack.push(i);
        }
        System.out.println(stack);
        int index = -1;
        int swapNum = -1;
        for (int i = 0; i < list.size(); i++) {
            if (!stack.isEmpty()) {
                int j = stack.pollLast();
                if (j != list.get(i)) {
                    index = i;
                    swapNum = j;
                    break;
                }
            } else {
                break;
            }
        }
        System.out.println(index);
        System.out.println(swapNum);
        if (index == -1) return dummy;
        for (int i = list.size() - 1; i >= 0; i--) {
            if (list.get(i) == swapNum) {
                int temp = list.get(index);
                list.set(index, swapNum);
                list.set(i, temp);
            }
        }
        System.out.println(list);
        int res = 0;
        for (int i : list) {
            res *= 10;
            res += i;
        }
        return res;
    }

    @Test
    public void msTest() {
        System.out.println(maximumSwap(115));
    }

    public int alternatingSubarray(int[] nums) {
        if (nums.length < 2) return 0;
        int res = 0;
        boolean isValid = false;
        int curLen = 0;
        int prev = nums[1] - nums[0];
        if (prev == 1) {
            isValid = true;
            curLen = 2;
            res = 2;
        }
        for (int i = 2; i < nums.length; i++) {
            int cur = nums[i] - nums[i - 1];
            if (cur == 1 || cur == -1) {
                if (isValid && (cur + prev == 0)) {
                    curLen++;
                    res = Math.max(res, curLen);
                } else if (cur == 1) {
                    curLen = 2;
                    res = Math.max(res, curLen);
                    isValid = true;
                } else {
                    isValid = false;
                    curLen = 0;
                }
            } else {
                isValid = false;
                curLen = 0;
            }
            prev = cur;
        }
        return res == 0 ? -1 : res;
    }

    @Test
    public void alterSub() {
        System.out.println(alternatingSubarray(new int[]{6, 7, 6, 5, 6}));
        calBits(10, 1);
    }

    public int sumIndicesWithKSetBits(List<Integer> nums, int k) {
        int sum = 0;
        for (int i = 0; i < nums.size(); i++) {
            if (calBits(i, k)) {
                sum += nums.get(i);
            }
        }
        return sum;
    }

    public boolean calBits(int num, int k) {
        int bitCnt = 0;
        while (num != 0) {
            if (num % 2 == 1) {
                bitCnt++;
                if (bitCnt > k) return false;
            }
            num /= 2;
        }
        return bitCnt == k;
    }

    public int maxNumberOfAlloys(int n, int k, int budget, List<List<Integer>> composition, List<Integer> stock, List<Integer> cost) {
        int max = 0;
        for (int i = 0; i < k; i++) {
            int maxNumber = calMax(composition.get(i), stock, cost, n, budget);
            System.out.println(maxNumber);
            max = Math.max(max, numberOfAlloys(n, budget, maxNumber, composition.get(i), stock, cost));
            System.out.println(max);
        }
        return max;
    }

    public int calMax(List<Integer> compose, List<Integer> stock, List<Integer> cost, int n, int budget) {
        int fee = 0;
        for (int i = 0; i < n; i++) {
            fee += compose.get(i) * cost.get(i);
        }
        int maxStock = 0;
        for (int i = 0; i < n; i++) {
            maxStock = Math.max(stock.get(i) / compose.get(i), maxStock);
        }
        return maxStock + budget / fee;
    }

    public int numberOfAlloys(int n, int budget, int right, List<Integer> compose, List<Integer> stock, List<Integer> cost) {
        int left = 0;
        while (left <= right) {
            int mid = left + (right - left) / 2;
            long costBudget = calCost(n, compose, stock, cost, mid);
            if (costBudget > budget) {
                right = mid - 1;
            } else {
                left = mid + 1;
            }
        }
        return right;
    }

    public long calCost(int n, List<Integer> compose, List<Integer> stock, List<Integer> cost, int cnt) {
        long costBudget = 0;
        for (int i = 0; i < n; i++) {
            long needCnt = (long) cnt * compose.get(i);
            int cur = stock.get(i);
            if (cur < needCnt) {
                costBudget += (needCnt - cur) * cost.get(i);
            }
        }
        return costBudget;
    }

    public boolean canMeasureWater(int jug1Capacity, int jug2Capacity, int targetCapacity) {
        if (jug1Capacity == targetCapacity || jug2Capacity == targetCapacity || jug1Capacity + jug2Capacity == targetCapacity)
            return true;
        if (jug1Capacity + jug2Capacity < targetCapacity || jug1Capacity == jug2Capacity) return false;
        return targetCapacity % (gcd(jug1Capacity, jug2Capacity)) == 0;
    }

    public int gcd(int x, int y) {
        int reminder = x % y;
        while (reminder != 0) {
            x = y;
            y = reminder;
            reminder = x % y;
        }
        return y;
    }

    @Test
    public void testFor() {
        try {
            Thread.sleep(1000);
        } catch (Exception e) {
            throw new RuntimeException("runtimeException");
        }
    }

    /**
     * 找到冠军 II
     */
    public int findChampion(int n, int[][] edges) {
        List<List<Integer>> adj = new ArrayList<>();
        int[] inDegrees = new int[n];
        for (int i = 0; i < n; i++) {
            adj.add(new ArrayList<>());
        }
        for (int[] edge : edges) {
            adj.get(edge[0]).add(edge[1]);
            inDegrees[edge[1]]++;
        }
        Deque<Integer> queue = new ArrayDeque<>();
        for (int i = 0; i < n; i++) {
            if (inDegrees[i] == 0) {
                queue.offer(i);
            }
        }
        if (queue.size() != 1) {
            return -1;
        }
        List<Integer> list = new ArrayList<>();
        return queue.peek();
    }

    public int[] findOriginalArray(int[] changed) {
        int len = changed.length;
        if (len % 2 != 0) return new int[]{};
        Arrays.sort(changed);
        Map<Integer, Integer> map = new HashMap<>();
        int l = len / 2;
        int[] res = new int[l];
        int index = 0;
        for (int i = 0; i < len; i++) {
            int original = changed[i];
            int change = original * 2;
            if (!map.containsKey(original)) {
                map.put(change, map.getOrDefault(change, 0) + 1);
            } else {
                int count = map.get(original);
                if (count == 1) {
                    map.remove(original);
                } else {
                    map.put(original, map.get(original) - 1);
                }
                res[index++] = original / 2;
            }
        }
        if (!map.isEmpty()) return new int[]{};
        return res;
    }

    private int getOriginal(int num) {
        if (num % 2 != 0) return -1;
        else return num / 2;
    }

    public int minSkips(int[] dist, int speed, int hoursBefore) {
        int n = dist.length;
        final double EPS = 1e-7;
        double[][] dp = new double[n + 1][n + 1];
        for (int i = 0; i <= n; i++) {
            Arrays.fill(dp[i], Double.MAX_VALUE);
        }
        dp[0][0] = 0;
        for (int i = 1; i <= n; i++) {
            for (int j = 0; j <= i; j++) {
                if (j != i) {
                    dp[i][j] = Math.min(Math.ceil(dp[i - 1][j] + (double) dist[i - 1] / speed - EPS), dp[i][j]);
                }
                if (j != 0) {
                    dp[i][j] = Math.min(dp[i][j], dp[i - 1][j - 1] + (double) dist[i - 1] / speed);
                }
            }
        }
        for (int j = 0; j <= n; j++) {
            if (dp[n][j] < hoursBefore + EPS) {
                return j;
            }
        }
        return -1;
    }


    List<List<Integer>> resCombinationSum3 = new ArrayList<>();
    LinkedList<Integer> pathCombinationSum3 = new LinkedList<>();

    public List<List<Integer>> combinationSum3(int k, int n) {
        backtrackingCombinationSum3(0, n, k);
        return resCombinationSum3;
    }

    private void backtrackingCombinationSum3(int index, int n, int k) {
        if (k == 0) {
            if (n == 0) resCombinationSum3.add(new ArrayList<>(pathCombinationSum3));
            return;
        }
        for (int i = index; i < 10; i++) {
            if (n < i) break;
            pathCombinationSum3.add(i);
            backtrackingCombinationSum3(i + 1, n - i, k - 1);
            pathCombinationSum3.removeLast();
        }
    }

    List<List<Integer>> resCombinationSum4 = new ArrayList<>();
    LinkedList<Integer> pathCombinationSum4 = new LinkedList<>();

    public int combinationSum4(int[] nums, int target) {
        int[] dp = new int[target + 1];
        dp[0] = 1;
        Arrays.sort(nums);
        for (int i = 1; i <= target; i++) {
            for (int j = 0; j < nums.length; j++) {
                if (nums[j] > i) break;
                dp[i] += dp[i - nums[j]];
            }
        }
        return dp[target];
    }

    private void backtrackingCombinationSum4(int target, int[] nums) {
        if (target == 0) {
            resCombinationSum4.add(new ArrayList<>(pathCombinationSum4));
            return;
        }
        for (int i = 0; i < nums.length; i++) {
            if (nums[i] > target) break;
            pathCombinationSum4.add(nums[i]);
            backtrackingCombinationSum4(target - nums[i], nums);
            pathCombinationSum4.removeLast();
        }
    }

    public int[] findColumnWidth(int[][] grid) {
        int m = grid.length;
        int n = grid[0].length;
        int[] res = new int[n];
        for (int i = 0; i < n; i++) {
            int width = Integer.MIN_VALUE;
            for (int j = 0; j < m; j++) {
                width = Math.max(width, calculateLen(grid[j][i]));
            }
            res[i] = width;
        }
        return res;
    }

    private int calculateLen(int num) {
        if (num == 0) return 1;
        int res = 0;
        if (num < 0) {
            res++;
            num = -num;
        }
        while (num != 0) {
            num /= 10;
            res++;
        }
        return res;
    }

    public int minimumRounds(int[] tasks) {
        int n = tasks.length;
        int[] dp = new int[n + 1];
        getDp(n, dp);
        Arrays.sort(tasks);
        int num = tasks[0], index = 0, count = 0, res = 0;
        while (index < n) {
            if (tasks[index] == num) {
                count++;
            } else {
                if (count == 1 || index == (n - 1)) return -1;
                res += dp[count];
                count = 1;
                num = tasks[index];
            }
            index++;
        }
        if (count == 1) return -1;
        res += dp[count];
        return res;
    }

    private void getDp(int n, int[] dp) {
        if (n <= 3) {
            Arrays.fill(dp, 1);
            return;
        }
        dp[1] = 1;
        dp[2] = 1;
        dp[3] = 1;
        for (int i = 4; i <= n; i++) {
            dp[i] = Math.min(dp[i - 2], dp[i - 3]) + 1;
        }
    }

    public int longestEqualSubarray(List<Integer> nums, int k) {
        int n = nums.size();
        HashMap<Integer, List<Integer>> pos = new HashMap<>();
        for (int i = 0; i < n; i++) {
            pos.computeIfAbsent(nums.get(i), x -> new ArrayList<>()).add(i);
        }
        int ans = 0;
        for (List<Integer> value : pos.values()) {
            // for循环里递增的一般是滑动窗口的右指针
            for (int i = 0, j = 0; i < value.size(); i++) {
                while (value.get(i) - value.get(j) - (i - j) > k) {
                    j++;
                }
                ans = Math.max(ans, i - j + 1);
            }
        }
        return ans;
    }

    public int[] mostCompetitive(int[] nums, int k) {
        Deque<Integer> stack = new ArrayDeque<>();
        int n = nums.length;
        for (int i = 0; i < n; i++) {
            while (!stack.isEmpty() && stack.peek() > nums[i] && (stack.size() + (n - i) > k)) {
                stack.pop();
            }
            stack.push(nums[i]);
        }
        int[] res = new int[k];
        for (int i = 0; i < k; i++) {
            res[i] = stack.pollLast();
        }
        return res;

    }

    public int[] findIndices(int[] nums, int indexDifference, int valueDifference) {
        int n = nums.length;
        int right = 0;
        for (int left = 0; left < n; left++) {
            while (right < n) {
                if ((right - left >= indexDifference) && (Math.abs(nums[right] - nums[left]) >= valueDifference)) {
                    return new int[]{left, right};
                }
                right++;
            }
            right = left;
        }
        return new int[]{-1, -1};
    }

    @Test
    public void testForMostCompetitive() {
        mostCompetitive(new int[]{3, 5, 2, 6}, 2);
    }

    ArrayList<List<Integer>> resMissingRolls = new ArrayList<>();
    LinkedList<Integer> pathMissingRolls = new LinkedList<>();
    int nMissingRolls;

    public int[] missingRolls(int[] rolls, int mean, int n) {
        int m = rolls.length;
        int target = (m + n) * mean - Arrays.stream(rolls).sum();
        if (target < n || target > 6 * n) return new int[]{};
        int quotient = target / n;
        int remainder = target % n;
        int[] res = new int[n];

        for (int i = 0; i < n; i++) {
            res[i] = quotient + ((i < remainder) ? 1 : 0);
        }
        return res;
    }


    public void backtrackingMissingRolls(int target) {
        if (target < 0 || pathMissingRolls.size() > nMissingRolls) {
            return;
        }
        if (target == 0 && pathMissingRolls.size() == nMissingRolls) {
            resMissingRolls.add(new ArrayList<>(pathMissingRolls));
        }
        for (int i = 1; i <= 6; i++) {
            if (target - i < 0) break;
            pathMissingRolls.add(i);
            backtrackingMissingRolls(target - i);
            pathMissingRolls.removeLast();
        }
    }

    public List<Integer> findPeaks(int[] mountain) {
        List<Integer> res = new ArrayList<>();
        for (int i = 1; i < mountain.length - 1; i++) {
            if (mountain[i] > mountain[i - 1] && mountain[i] > mountain[i + 1]) {
                res.add(i++);
            }
        }
        return res;
    }

    public int maximumLength(String s) {
        int len = s.length();
        int res = -1;
        List<Integer>[] chs = new List[26];
        for (int i = 0; i < 26; i++) {
            chs[i] = new ArrayList<>();
        }
        int cnt = 0;
        for (int i = 0; i < len; i++) {
            cnt++;
            if (i == len - 1 || s.charAt(i) != s.charAt(i + 1)) {
                int ch = s.charAt(i) - 'a';
                chs[ch].add(cnt);
                cnt = 0;

                for (int j = chs[ch].size() - 1; j > 0; j--) {
                    if (chs[ch].get(j) > chs[ch].get(j - 1)) {
                        Collections.swap(chs[ch], j, j - 1);
                    } else {
                        break;
                    }
                }
                if (chs[ch].size() > 3) {
                    chs[ch].remove(chs[ch].size() - 1);
                }
            }
        }

        for (int i = 0; i < 26; i++) {
            if (chs[i].size() > 0 && chs[i].get(0) > 2) {
                res = Math.max(res, chs[i].get(0) - 2);
            }
            if (chs[i].size() > 1 && chs[i].get(0) > 1) {
                res = Math.max(res, Math.min(chs[i].get(0) - 1, chs[i].get(1)));
            }
            if (chs[i].size() > 2) {
                res = Math.max(res, chs[i].get(2));
            }
        }
        return res;

    }

    public int[] findMissingAndRepeatedValues(int[][] grid) {
        int n = grid.length;
        Set<Integer> set = new HashSet<>();
        for (int i = 1; i <= n * n; i++) {
            set.add(i);
        }
        int[] res = new int[2];
        for (int[] ints : grid) {
            for (int j = 0; j < n; j++) {
                if (set.contains(ints[j])) {
                    set.remove(ints[j]);
                } else {
                    res[0] = ints[j];
                }
            }
        }
        List<Integer> list = new ArrayList<>(set);
        res[1] = list.get(0);
        return res;
    }

    int limit;
    int mDistributeCandies;
    int resDistributeCandies;
    LinkedList<Integer> pathDistributeCandies = new LinkedList<>();


    public void backtrackingDistributeCandies(int n) {
        if (pathDistributeCandies.size() == mDistributeCandies) {
            if (n == 0) {
                resDistributeCandies++;
            }
            return;
        }
        for (int i = 0; i <= limit; i++) {
            if (n < i) break;
            pathDistributeCandies.add(i);
            backtrackingDistributeCandies(n - i);
            pathDistributeCandies.removeLast();
        }
    }


    public int distributeCandies(int[] candyType) {
        int n = candyType.length;
        Set<Integer> set = new HashSet<>();
        for (int i = 0; i < n; i++) {
            set.add(candyType[i]);
        }
        return Math.min(n / 2, set.size());
    }

    public int[] distributeCandies(int candies, int num_people) {
        int[] res = new int[num_people];
        int num = 1;
        while (candies >= num) {
            res[(num - 1) % num_people] += num;
            candies -= num;
            num++;
        }
        res[(num - 1) % num_people] += candies;
        return res;
    }

    public long minimumSteps(String s) {
        int n = s.length();
        long res = 0;
        int sum = 0;
        for (int i = 0; i < n; i++) {
            if (s.charAt(i) == '1') sum++;
            else res += sum;
        }
        return res;
    }

    public int maxOperations(int[] nums) {
        int n = nums.length;
        int cnt = nums[0] + nums[1];
        int res = 1;
        for (int i = 2; i < n; i += 2) {
            if (i != (n - 1) && nums[i] + nums[i + 1] == cnt) res++;
            else break;
        }
        return res;
    }

    public boolean isMatch(String s, String p) {
        int m = s.length();
        int n = p.length();
        boolean[][] dp = new boolean[m + 1][n + 1];
        dp[0][0] = true;
        for (int i = 0; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                if (p.charAt(j - 1) == '*') {
                    dp[i][j] = dp[i][j - 2];
                    if (matches(s, p, i, j - 1)) {
                        dp[i][j] = dp[j][j] || dp[i - 1][j];
                    }
                } else {
                    if (matches(s, p, i, j)) {
                        dp[i][j] = dp[i - 1][j - 1];
                    }
                }
            }
        }
        return dp[m][n];
    }

    private boolean matches(String s, String p, int i, int j) {
        if (i == 0) return false;
        if (p.charAt(j - 1) == '.') return true;
        return s.charAt(i - 1) == p.charAt(j - 1);
    }

    public int maximumBeauty(int[] nums, int k) {
        Arrays.sort(nums);
        int n = nums.length, res = 0;
        for (int i = 0, j = 0; i < n; i++) {
            while (nums[i] - 2 * k > nums[j]) {
                j++;
            }
            res = Math.max(res, i - j + 1);
        }
        return res;
    }

    public int findLUSlength(String a, String b) {
        return a.equals(b) ? -1 : Math.max(a.length(), b.length());
    }

    public int findLUSlength(String[] strs) {
        int n = strs.length;
        int ans = -1;
        for (int i = 0; i < n; i++) {
            boolean check = true;
            for (int j = 0; j < n; j++) {
                if (i != j && isSubSeq(strs[i], strs[j])) {
                    check = false;
                    break;
                }
            }
            if (check) {
                ans = Math.max(ans, strs[i].length());
            }
        }
        return ans;
    }

    private boolean isSubSeq(String s, String t) {
        int ps = 0, pt = 0;
        while (ps < s.length() && pt < t.length()) {
            if (s.charAt(ps) == t.charAt(pt)) {
                ps++;
            }
            pt++;
        }
        return ps == s.length();
    }

    public String discountPrices(String sentence, int discount) {
        StringBuilder res = new StringBuilder();
        for (String s : sentence.split(" ")) {
            long price = isPrice(s);
            if (price != -1) {
                res.append("$").append(discountPrice(price, discount)).append(" ");
            } else {
                res.append(s).append(" ");
            }
        }
        return res.toString().trim();
    }

    private long isPrice(String priceStr) {
        int n = priceStr.length();
        if (n < 2 || priceStr.charAt(0) != '$' || priceStr.charAt(1) == '0') return -1;
        for (int i = 1; i < n; i++) {
            if (priceStr.charAt(i) < '0' || priceStr.charAt(i) > '9') return -1;
        }
        return Long.parseLong(priceStr.substring(1));
    }

    private String discountPrice(long price, int discount) {
        double res = price * (1 - discount / 100.0);
        return String.format("$%.2f", res);
    }

    @Test
    public void discountPricess() {
        double v = 100 * 8 * 0.01;
        System.out.println(v);
    }

    public int[] nextGreaterElements(int[] nums) {
        int n = nums.length;
        int[] res = new int[n];
        Arrays.fill(res, -1);
        Deque<Integer> stack = new ArrayDeque<>();
        for (int i = 0; i < 2 * n - 1; i++) {
            while (!stack.isEmpty() && nums[stack.peek() % n] < nums[i % n]) {
                res[stack.pop() % n] = nums[i % n];
            }
            stack.push(i);
        }
        return res;
    }

    public int countArrangement(int n) {
        int[] memo = new int[1 << n];
        Arrays.fill(memo, -1);
        return dfsCountArrangement(0, n, memo);
    }

    public int dfsCountArrangement(int s, int n, int[] memo) {
        if (s == (1 << n) - 1) {
            return 1;
        }
        if (memo[s] != -1) {
            return memo[s];
        }
        int res = 0;
        int i = Integer.bitCount(s) + 1;
        for (int j = 1; j <= n; j++) {
            if ((s >> (j - 1) & 1) == 0 && (i % j == 0 || j % i == 0)) {
                res += dfsCountArrangement(s | (1 << (j - 1)), n, memo);
            }
        }
        memo[s] = res;
        return res;
    }

    public int specialPerm(int[] nums) {
        int n = nums.length;
        long[][] memo = new long[1 << n][n];
        for (long[] longs : memo) {
            Arrays.fill(longs, -1);
        }
        long res = 0;
        for (int i = 0; i < n; i++) {
            res += dfsSpecialPerm(1 << i, n, memo, i, nums);
        }
        return (int) (res % 1000000007);
    }

    public long dfsSpecialPerm(int s, int n, long[][] memo, int j, int[] nums) {
        if (s == (1 << n) - 1) {
            return 1;
        }
        if (memo[s][j] != -1) {
            return memo[s][j];
        }
        long res = 0;
        for (int i = 0; i < n; i++) {
            if (((s >> i) & 1) == 0 && (nums[j] % nums[i] == 0 || nums[i] % nums[j] == 0)) {
                res += dfsSpecialPerm(s | (1 << i), n, memo, i, nums);
            }
        }
        memo[s][j] = res;
        return res;
    }

    public String smallestString(String s) {
        char[] chars = new char[]{'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'};
        int n = s.length();
        StringBuilder res = new StringBuilder();
        int left = -1, right = -1;
        for (int i = 0; i < n; i++) {
            if (s.charAt(i) != 'a') {
                left = i;
                break;
            }
            res.append('a');
        }
        if (left == -1) {
            return s.substring(0, n - 1) + 'z';
        }
        for (int i = left; i < n; i++) {
            if (s.charAt(i) != 'a') {
                res.append(chars[(s.charAt(i) - 'a') - 1]);
            } else {
                right = i;
                break;
            }
        }
        if (right != -1) {
            for (int i = right; i < n; i++) {
                res.append(s.charAt(i));
            }
        }
        return res.toString();
    }

    @Test
    public void smallestStringTest() {
        System.out.println('b' - 1);
    }

    public int pivotIndex(int[] nums) {
        int sum = Arrays.stream(nums).sum();
        int leftCnt = 0;
        for (int i = 0; i < nums.length; i++) {
            if ((leftCnt * 2) == (sum - nums[i])) {
                return i;
            }
            leftCnt += nums[i];
        }
        return -1;
    }

    public int incremovableSubarrayCount(int[] nums) {
        int n = nums.length;
        int res = 0;
        int l = 1;
        while (l < n && nums[l] > nums[l - 1]) {
            l++;
        }
        res += l + (l < n ? 1 : 0);
        for (int r = n - 2; r >= 0; r--) {
            while (l > 0 && nums[r + 1] <= nums[l - 1]) {
                l--;
            }
            res += l + (l <= r ? 1 : 0);
            if (nums[r] >= nums[r + 1]) {
                break;
            }
        }
        return res;
    }

    public boolean isIncreasing(int[] nums, int l, int r) {
        for (int i = 1; i < nums.length; i++) {
            if (i >= l && i <= r + 1) {
                continue;
            }
            if (nums[i] <= nums[i - 1]) {
                return false;
            }
        }
        if (l - 1 >= 0 && r + 1 < nums.length && nums[r + 1] <= nums[l - 1]) {
            return false;
        }
        return true;
    }

    public int minimumDistance(int[][] points) {
        TreeMap<Integer, Integer> xs = new TreeMap<>();
        TreeMap<Integer, Integer> ys = new TreeMap<>();
        // 将点绕原点顺时针旋转45度后并扩大根号2倍后的坐标值
        for (int[] point : points) {
            xs.merge(point[0] + point[1], 1, Integer::sum);
            ys.merge(point[1] - point[0], 1, Integer::sum);
        }
        int ans = Integer.MAX_VALUE;
        for (int[] p : points) {
            int x = p[0] + p[1];
            int y = p[1] - p[0];
            // 有可能有相同的点，所以要判断，value表示在该位置的点的个数
            if (xs.get(x) == 1) xs.remove(x);
            else xs.merge(x, -1, Integer::sum);
            if (ys.get(y) == 1) ys.remove(y);
            else ys.merge(y, -1, Integer::sum);

            int dx = xs.lastKey() - xs.firstKey();
            int dy = ys.lastKey() - ys.firstKey();
            ans = Math.min(ans, Math.max(dx, dy));

            // 枚举完该点后恢复其个数
            xs.merge(x, 1, Integer::sum);
            ys.merge(y, 1, Integer::sum);
        }
        return ans;
    }

    public boolean canSortArray(int[] nums) {
        int n = nums.length;
        int preMax = 0;
        // 外循环没有递增逻辑
        for (int i = 0; i < n; ) {
            int curMax = 0;
            int ones = Integer.bitCount(nums[i]);
            while (i < n && Integer.bitCount(nums[i]) == ones) {
                if (nums[i] < preMax) {
                    return false;
                }
                curMax = Math.max(curMax, nums[i++]);
            }
            preMax = curMax;
        }
        return true;
    }

    public int[] numberGame(int[] nums) {
        Arrays.sort(nums);
        int n = nums.length;
        int[] arr = new int[n];
        for (int i = 0; i < n; i += 2) {
            arr[i] = nums[i + 1];
            arr[i + 1] = nums[i];
        }
        return arr;
    }

    public int maxIncreaseKeepingSkyline(int[][] grid) {
        int n = grid.length;
        int[] rowMax = new int[n];
        int[] colMax = new int[n];
        for (int i = 0; i < n; i++) {
            rowMax[i] = 0;
            for (int j = 0; j < n; j++) {
                rowMax[i] = Math.max(rowMax[i], grid[i][j]);
            }
        }
        for (int i = 0; i < n; i++) {
            colMax[i] = 0;
            for (int j = 0; j < n; j++) {
                colMax[i] = Math.max(colMax[i], grid[j][i]);
            }
        }
        int res = 0;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                res += Math.min(rowMax[i], colMax[j]) - grid[i][j];
            }
        }
        return res;

    }

    public String intToRoman(int num) {
        char[] chars = new char[]{'I', 'V', 'X', 'L', 'C', 'D', 'M'};
        List<Integer> nums = getNums(num);
        StringBuilder res = new StringBuilder();
        for (int i = 0; i < nums.size(); i++) {
            int n = nums.get(i);
            if (n > 0 && n < 4) {
                for (int j = 0; j < n; j++) {
                    res.append(chars[2 * i]);
                }
            } else if (n > 4 && n < 9) {
                for (int j = 5; j < n; j++) {
                    res.append(chars[2 * i]);
                }
                res.append(chars[2 * i + 1]);
            } else if (n == 4) {
                res.append(chars[2 * i + 1]).append(chars[2 * i]);
            } else if (n == 9) {
                res.append(chars[2 * (i + 1)]).append(chars[2 * i]);
            }
        }
        return res.reverse().toString();
    }

    private List<Integer> getNums(int num) {
        List<Integer> list = new ArrayList<>();
        while (num != 0) {
            list.add(num % 10);
            num /= 10;
        }
        return list;
    }

    public int romanToInt(String s) {
        int n = s.length();
        char[] chars = new char[]{'I', 'V', 'X', 'L', 'C', 'D', 'M'};
        int[] ints = new int[]{1, 5, 10, 50, 100, 500, 1000};
        Map<Character, Integer> map = new HashMap<>();
        for (int i = 0; i < chars.length; i++) {
            map.put(chars[i], ints[i]);
        }
        int res = 0;
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            if (c == 'I' && i < n - 1 && s.charAt(i + 1) == 'V') {
                res += 4;
                i++;
            } else if (c == 'I' && i < n - 1 && s.charAt(i + 1) == 'X') {
                res += 9;
                i++;
            } else if (c == 'X' && i < n - 1 && s.charAt(i + 1) == 'L') {
                res += 40;
                i++;
            } else if (c == 'X' && i < n - 1 && s.charAt(i + 1) == 'C') {
                res += 90;
                i++;
            } else if (c == 'C' && i < n - 1 && s.charAt(i + 1) == 'D') {
                res += 400;
                i++;
            } else if (c == 'C' && i < n - 1 && s.charAt(i + 1) == 'M') {
                res += 900;
                i++;
            } else {
                res += map.get(c);
            }
        }
        return res;
    }

    public int threeSumClosest(int[] nums, int target) {
        int n = nums.length;
        Arrays.sort(nums);
        int res = nums[0] + nums[1] + nums[2];
        for (int i = 0; i < n - 2; i++) {
            int min = nums[i] + nums[i + 1] + nums[i + 2];
            int max = nums[i] + nums[n - 2] + nums[n - 1];
            if (min > target) {
                if (Math.abs(min - target) < Math.abs(res - target)) {
                    res = min;
                }
                break;
            }
            if (max < res) {
                if (Math.abs(max - target) < Math.abs(res - target)) {
                    res = max;
                }
                continue;
            }
            int left = i + 1, right = n - 1;
            while (left < right) {
                int sum = nums[i] + nums[left] + nums[right];
                if (sum < target) {
                    left++;
                } else if (sum > target) {
                    right--;
                } else {
                    return target;
                }
                if (Math.abs(sum - target) < Math.abs(res - target)) {
                    res = sum;
                }
            }
        }
        return res;
    }

    public ListNode mergeTwoLists(ListNode list1, ListNode list2) {
        if (list1 == null) {
            return list2;
        } else if (list2 == null) {
            return list1;
        } else if (list1.val > list2.val) {
            list2.next = mergeTwoLists(list1, list2.next);
            return list2;
        } else {
            list1.next = mergeTwoLists(list1.next, list2);
            return list1;
        }

    }

    public List<List<String>> accountsMerge(List<List<String>> accounts) {
        int n = accounts.size();
        Map<String, List<Integer>> emailToIdx = new HashMap<>();
        for (int i = 0; i < n; i++) {
            for (int j = 1; j < accounts.get(i).size(); j++) {
                emailToIdx.computeIfAbsent(accounts.get(i).get(j), x -> new ArrayList<>()).add(i);
            }
        }
        ArrayList<List<String>> ans = new ArrayList<>();
        boolean[] vis = new boolean[n];
        HashSet<String> emailSet = new HashSet<>();
        for (int i = 0; i < n; i++) {
            if (vis[i]) {
                continue;
            }
            emailSet.clear();
            accountsMergeDFS(accounts, i, emailToIdx, vis, emailSet);
            List<String> res = new ArrayList<>(emailSet);
            Collections.sort(res);
            res.add(0, accounts.get(i).get(0));
            ans.add(res);
        }
        return ans;
    }

    private void accountsMergeDFS(List<List<String>> accounts, int i, Map<String, List<Integer>> emailToIdx, boolean[] vis, Set<String> emailSet) {
        vis[i] = true;
        for (int j = 1; j < accounts.get(i).size(); j++) {
            String email = accounts.get(i).get(j);
            if (emailSet.contains(email)) {
                continue;
            }
            emailSet.add(email);
            for (int idx : emailToIdx.get(email)) {
                if (!vis[idx]) {
                    accountsMergeDFS(accounts, idx, emailToIdx, vis, emailSet);
                }
            }

        }
    }

    public int[] findIntersectionValues(int[] nums1, int[] nums2) {
        HashSet<Integer> set1 = Arrays.stream(nums1).boxed().collect(Collectors.toCollection(HashSet::new));
        HashSet<Integer> set2 = Arrays.stream(nums1).boxed().collect(Collectors.toCollection(HashSet::new));
        int[] res = new int[2];
        ArrayList<Integer> list = new ArrayList<>();
        for (int i : nums1) {
            res[0] += set2.contains(i) ? 1 : 0;
        }
        for (int i : nums2) {
            res[1] += set1.contains(i) ? 1 : 0;
        }
        return res;
    }

    public int numberOfSets(int n, int maxDistance, int[][] roads) {
        int[][] g = new int[n][n];
        for (int[] row : g) {
            // 防止加法溢出
            Arrays.fill(row, Integer.MAX_VALUE / 2);
        }
        for (int[] road : roads) {
            int x = road[0], y = road[1], wt = road[2];
            g[x][y] = Math.min(g[x][y], wt);
            g[y][x] = Math.min(g[y][x], wt);
        }
        int ans = 0;
        int[][] f = new int[n][n];
        next:
        for (int s = 0; s < (1 << n); s++) {
            for (int i = 0; i < n; i++) {
                // 如果该点存在，则转移数组
                if ((s >> i & 1) == 1) {
                    System.arraycopy(g[i], 0, f[i], 0, n);
                }
            }
            // Floyd算法
            for (int k = 0; k < n; k++) {
                if ((s >> k & 1) == 0) continue;
                for (int i = 0; i < n; i++) {
                    if ((s >> i & 1) == 0) continue;
                    // 只有两个点都在时，才计算最短距离
                    for (int j = 0; j < n; j++) {
                        f[i][j] = Math.min(f[i][j], f[i][k] + f[k][j]);
                    }
                }
            }

            for (int i = 0; i < n; i++) {
                if ((s >> i & 1) == 0) continue;
                for (int j = 0; j < i; j++) {
                    // 当两个点都在并且距离超过最大值时，此方案无效
                    if ((s >> j & 1) == 1 && f[i][j] > maxDistance) {
                        continue next;
                    }
                }
            }
            ans++;
        }
        return ans;
    }

    public int[] minimumTime(int n, int[][] edges, int[] disappear) {
        List<int[]>[] adj = new ArrayList[n];
        Arrays.setAll(adj, i -> new ArrayList<>());
        for (int[] edge : edges) {
            int u = edge[0];
            int v = edge[1];
            int wt = edge[2];
            adj[u].add(new int[]{v, wt});
            adj[v].add(new int[]{u, wt});
        }
        int[] dis = new int[n];
        Arrays.fill(dis, 1);
        dis[0] = 0;
        PriorityQueue<int[]> pq = new PriorityQueue<>(((o1, o2) -> o1[0] - o2[0]));
        pq.offer(new int[]{0, 0});
        while (!pq.isEmpty()) {
            int[] p = pq.poll();
            int dx = p[0];
            int x = p[1];
            if (dx > dis[x]) {
                continue;
            }
            for (int[] e : adj[x]) {
                int y = e[0];
                int wt = e[1];
                int newDis = wt + dx;
                if (disappear[y] > newDis && (dis[y] < 0 || dis[y] > newDis)) {
                    dis[y] = newDis;
                    pq.offer(new int[]{newDis, y});
                }
            }
        }
        return dis;
    }

    public int minimumLevels(int[] possible) {
        int sum = 0;
        for (int i : possible) {
            sum += i == 1 ? 1 : -1;
        }
        int cur;
        if (sum >= 0) {
            cur = sum / 2 + 1;
        } else {
            cur = (sum % 2 == 0) ? (sum / 2 + 1) : (sum / 2);
        }
        System.out.println(cur);
        int cnt = 0;
        for (int i = 0; i < possible.length - 1; i++) {
            if (possible[i] == 1) {
                cnt++;
            } else {
                cnt--;
            }
            if (cnt == cur) {
                return i + 1;
            }
        }
        return -1;
    }

    private void dfsMinimumTime(int u, Map<Integer, Map<Integer, Integer>> map, boolean[] visited, int[] disappear, int[] res, int preV) {
        if (!map.containsKey(u) || visited[u]) {
            return;
        }
        Map<Integer, Integer> adjMap = map.get(u);
        for (int v : adjMap.keySet()) {
            int cur = preV + adjMap.get(v);
            if (cur >= disappear[v]) {
                continue;
            }
            res[v] = Math.min(res[v], cur);
            visited[u] = true;
            dfsMinimumTime(v, map, visited, disappear, res, cur);
            visited[u] = false;
        }
    }

    @Test
    public void test0718() {
        PriorityQueue<Integer> queue = new PriorityQueue<>(((o1, o2) -> o2 - o1));
        queue.offer(1);
        queue.offer(2);
        System.out.println(queue.poll());
    }


}


class MyException extends RuntimeException {
    MyException() {

    }

    MyException(String msg) {
        super("我的异常： " + msg);
    }

    MyException(String msg, RuntimeException e) {
        super("我的异常： " + msg, e);
    }
}

