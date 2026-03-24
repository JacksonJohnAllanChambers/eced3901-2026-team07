/* SCHEDULER IMPLEMENTATION */

#include "schedule.h"

volatile uint32_t count = 0;
volatile uint32_t ucount = 0;

void scheduler_tick(void){
    count++;
    ucount++;
    // 60ms timing schedule
    if(ucount > 3000) ucount = 0; 
}