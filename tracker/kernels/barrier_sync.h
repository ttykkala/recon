/*
Copyright 2016 Tommi M. Tykkälä

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#pragma once

#include <multithreading.h>

typedef struct {
    pthread_mutex_t count_lock;
    pthread_cond_t ok_to_proceed;
    int count;
} barrier_t;

class barrierSync {
  public:
    barrierSync() {
    }
    void initialize(int nProcesses) {
        this->nProcesses = nProcesses;
        barrier.count = 0;
        pthread_mutex_init(&(barrier.count_lock), NULL);
        pthread_cond_init(&(barrier.ok_to_proceed), NULL);
        initialized = true;
        shuttingDown = false;
    }
    ~barrierSync() {
        if (initialized) {
            pthread_mutex_destroy(&(barrier.count_lock));
            pthread_cond_destroy(&(barrier.ok_to_proceed));
        }
    }
    void disable() {
        if (!initialized) return;
        pthread_mutex_lock(&(barrier.count_lock));
        shuttingDown = true;
        pthread_cond_broadcast(&(barrier.ok_to_proceed));
        pthread_mutex_unlock(&(barrier.count_lock));
        initialized = false;
    }
    void sync() {
        if (!initialized) return;
        pthread_mutex_lock(&(barrier.count_lock));
        barrier.count++;
        if (barrier.count == nProcesses) {
            barrier.count = 0;
            pthread_cond_broadcast(&(barrier.ok_to_proceed));
        }
        else if (!shuttingDown){
            while (pthread_cond_wait(&(barrier.ok_to_proceed),
                                     &(barrier.count_lock)) != 0);
        }
        pthread_mutex_unlock(&(barrier.count_lock));
    }
private:
    barrier_t barrier;
    int nProcesses;
    bool initialized;
    bool shuttingDown;
};

