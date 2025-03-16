#ifndef BASE_SERVICE_HPP
#define BASE_SERVICE_HPP

#include <vector>
#include <fstream>

namespace Amtal {

template<typename V>
class Service_Listener {

public:

    // add event to a serivce
    virtual void add_process(V &data) = 0;

    // remove an event from a service
    virtual void remove_process(V &data) = 0;

    // process an event update to the service
    virtual void update_process(V& data) = 0;

};

template<typename K, typename V>
class Service {

public:

    // get data from a service
    virtual V& get_data(K key) = 0;
    
    // connectors invoke for new or updated data
    virtual void on_message(V &data) = 0;

    // add a listener to a service for callbacks on add, remove, and update events
    virtual void add_listener(Service_Listener<V> *listener) = 0;

    // get all listeners on the service
    virtual const std::vector<Service_Listener<V>*>& get_listeners() const = 0;

};

template<typename V>
class Connector {

public:

    // publish data to connector S
    virtual void publish(V &data) = 0;

};

}

#endif 