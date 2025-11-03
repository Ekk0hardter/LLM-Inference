
using namespace std;
// 哈希表的访问和修改的时间复杂度是O(1),可以使用哈希表存储key value的映射关系
// 可以使用队列来维护key的有序性（每次在队头插入数据，在队尾删除数据），但涉及到中间节点的
// 删除，在队列中无法在O(1)时间内完成，因此想到了双向链表
#include <unordered_map>

struct Node
{
    int key, val;
    Node *prev, *next;
    Node() : key(0), val(0), prev(nullptr), next(nullptr) {}
    Node(int _key, int _val) : key(_key), val(_val), prev(nullptr), next(nullptr) {}
};

class LRUCache
{
public:
    // 定义双向链表的虚拟头尾节点
    Node *head, *tail;
    // 定义哈希表
    unordered_map<int, Node *> mp;
    // 记录哈希表容量大小和已使用的大小
    int capacity, size;
    // 构造函数
    LRUCache(int _capacity) : capacity(_capacity), size(0)
    {
        // 创建虚拟的头尾节点
        head = new Node();
        tail = new Node();
        // 让头尾节点进行链接
        head->next = tail;
        tail->prev = head;
    }

    // 从链表中删除节点
    void removeNode(Node *node)
    {
        node->prev->next = node->next; // 让当前节点的上一个节点的next指向当前节点的下一个节点
        node->next->prev = node->prev; // 让当前节点的下一个节点的prev指向当前节点的上一个节点
    }
    // 插入节点到链表头
    void addNodeToHead(Node *node)
    {
        node->prev = head;
        head->next = node;

        node->next = head->next;
        // 让头结点的下一个节点的prev指向该节点
        head->next->prev = node;
        // 让头结点的next指向当前节点
        
    }

    // get方法
    int get(int key)
    {
        if (!mp.count(key))
            return -1; // 若哈希表中不存在该key，返回-1
        // 若存在，获取一下对应的节点
        Node *node = mp[key];
        // 将该节点从链表中删除，并插入到头结点后面（意为最新）
        removeNode(node);
        addNodeToHead(node);
        return node->val; // 返回该节点的值
    }

    // put方法
    void put(int key, int value)
    {
        // 若哈希表中存在该key
        if (mp.count(key))
        {
            Node *node = mp[key]; // 新建一个node节点，拿到该node
            node->val = value;    // 更新该节点的值
            // 将该节点从链表中删除，并插入到头结点后面（意为最新）
            removeNode(node);
            addNodeToHead(node);
        }
        else
        {
            // 若哈希表中不存在该key
            // 若当前已使用的大小等于容量大小，需要从链表中删除最后一个节点
            if (size == capacity)
            {
                Node *remove = tail->prev; // 拿到链表中的最后一个节点
                mp.erase(remove->key);     // 将该节点从哈希表中删除
                removeNode(remove);        // 将该节点从链表中删除
                size--;
            }
            // 若还有容量
            // 新建一个节点并插入到链表头
            Node *node = new Node(key, value);
            addNodeToHead(node); // 插入到链表头
            mp[key] = node;      // 将该节点插入到哈希表中
            size++;
        }
    }
};

