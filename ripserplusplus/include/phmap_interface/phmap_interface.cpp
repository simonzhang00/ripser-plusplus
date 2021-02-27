#include <iostream>
#include <string>
#include <parallel_hashmap/phmap.h>

using phmap::parallel_flat_hash_map;

parallel_flat_hash_map<int64_t, int64_t> hash_map;

phmap::parallel_flat_hash_map<int64_t,int64_t>::iterator it;

void phmap_put(int64_t key, int64_t value){
    hash_map[key]= value;
}
int64_t phmap_get_value(int64_t key){
    auto pair= hash_map.find(key);
    if(pair !=hash_map.end()){
        return pair->second;
    }
    return -1;
}

void phmap_clear(){
    hash_map.clear();
}