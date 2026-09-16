/* SCHEDULER - TEAM 7*/

#ifndef SCHEDULE_H
#define SCHEDULE_H

// includes

#include <stdint.h>

// define counter variables

extern volatile uint32_t count;
extern volatile uint32_t ucount;
extern volatile uint32_t ccount;

// define counter functions

void scheduler_tick(void);

#endif