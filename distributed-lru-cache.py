from collections import OrderedDict
from threading import Lock
import time
from typing import Any, Optional, Dict, List
import hashlib

class CacheNode:
    """Represents a single cache node in the distributed system"""
    def __init__(self, capacity: int, node_id: str):
        self.capacity = capacity
        self.node_id = node_id
        self.cache: OrderedDict = OrderedDict()
        self.lock = Lock()
        self.next_node: Optional['CacheNode'] = None
        
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache with LRU update"""
        with self.lock:
            if key in self.cache:
                value = self.cache.pop(key)
                self.cache[key] = value
                return value
            return None
            
    def put(self, key: str, value: Any) -> None:
        """Put value in cache with LRU eviction"""
        with self.lock:
            if key in self.cache:
                self.cache.pop(key)
            elif len(self.cache) >= self.capacity:
                # Evict least recently used item
                self.cache.popitem(last=False)
            self.cache[key] = value

class DistributedLRUCache:
    """Distributed LRU Cache with consistent hashing"""
    def __init__(self, total_nodes: int, capacity_per_node: int):
        self.total_nodes = total_nodes
        self.capacity_per_node = capacity_per_node
        self.nodes: Dict[str, CacheNode] = {}
        self.virtual_nodes = 100  # Number of virtual nodes per physical node
        self.hash_ring: List[tuple[int, str]] = []  # (hash_value, node_id)
        
        # Initialize cache nodes and hash ring
        self._initialize_nodes()
        
    def _initialize_nodes(self) -> None:
        """Initialize cache nodes and build consistent hash ring"""
        # Create physical nodes
        for i in range(self.total_nodes):
            node_id = f"node_{i}"
            self.nodes[node_id] = CacheNode(self.capacity_per_node, node_id)
            
        # Create virtual nodes and build hash ring
        for node_id in self.nodes:
            for i in range(self.virtual_nodes):
                virtual_node_id = f"{node_id}_{i}"
                hash_value = self._get_hash(virtual_node_id)
                self.hash_ring.append((hash_value, node_id))
                
        # Sort hash ring
        self.hash_ring.sort()
        
        # Link nodes in a ring for replication
        node_ids = list(self.nodes.keys())
        for i in range(len(node_ids)):
            current_node = self.nodes[node_ids[i]]
            next_node = self.nodes[node_ids[(i + 1) % len(node_ids)]]
            current_node.next_node = next_node
            
    def _get_hash(self, key: str) -> int:
        """Generate hash value for key"""
        return int(hashlib.md5(key.encode()).hexdigest(), 16)
        
    def _get_node(self, key: str) -> CacheNode:
        """Find responsible node for key using consistent hashing"""
        hash_value = self._get_hash(key)
        
        # Binary search to find the next largest hash value
        left, right = 0, len(self.hash_ring)
        while left < right:
            mid = (left + right) % len(self.hash_ring)
            if self.hash_ring[mid][0] < hash_value:
                left = mid + 1
            else:
                right = mid
                
        # If we're past the last node, wrap around to the first
        index = left % len(self.hash_ring)
        node_id = self.hash_ring[index][1]
        return self.nodes[node_id]
        
    def get(self, key: str) -> Optional[Any]:
        """Get value from distributed cache"""
        node = self._get_node(key)
        value = node.get(key)
        
        if value is None and node.next_node:
            # Try backup node if primary doesn't have the value
            value = node.next_node.get(key)
            if value is not None:
                # Replicate back to primary node
                node.put(key, value)
                
        return value
        
    def put(self, key: str, value: Any) -> None:
        """Put value in distributed cache"""
        node = self._get_node(key)
        node.put(key, value)
        
        # Replicate to next node for fault tolerance
        if node.next_node:
            node.next_node.put(key, value)
            
    def remove_node(self, node_id: str) -> None:
        """Remove a node from the cluster (for handling node failures)"""
        if node_id not in self.nodes:
            return
            
        # Remove node and its virtual nodes from hash ring
        self.hash_ring = [(h, n) for h, n in self.hash_ring if n != node_id]
        
        # Update node links
        for node in self.nodes.values():
            if node.next_node and node.next_node.node_id == node_id:
                node.next_node = self.nodes[node_id].next_node
                
        del self.nodes[node_id]
        
    def add_node(self, node_id: str) -> None:
        """Add a new node to the cluster"""
        if node_id in self.nodes:
            return
            
        # Create new node
        new_node = CacheNode(self.capacity_per_node, node_id)
        self.nodes[node_id] = new_node
        
        # Add virtual nodes to hash ring
        for i in range(self.virtual_nodes):
            virtual_node_id = f"{node_id}_{i}"
            hash_value = self._get_hash(virtual_node_id)
            self.hash_ring.append((hash_value, node_id))
            
        # Sort hash ring
        self.hash_ring.sort()
        
        # Update node links
        keys = list(self.nodes.keys())
        idx = keys.index(node_id)
        prev_node = self.nodes[keys[idx - 1]]
        next_node = prev_node.next_node
        
        prev_node.next_node = new_node
        new_node.next_node = next_node
        
        # Rebalance data
        self._rebalance_data(new_node)
        
    def _rebalance_data(self, new_node: CacheNode) -> None:
        """Rebalance data when adding a new node"""
        for node in self.nodes.values():
            if node == new_node:
                continue
                
            keys_to_move = []
            for key in node.cache:
                if self._get_node(key) == new_node:
                    keys_to_move.append(key)
                    
            for key in keys_to_move:
                value = node.cache[key]
                new_node.put(key, value)
                node.cache.pop(key)
