/* UART MODULE */

#include "uart.h"
#include <avr/interrupt.h>

// TX ring buffer
#define TX_BUF_SIZE 32
static volatile char tx_buf[TX_BUF_SIZE];
static volatile uint8_t tx_head = 0, tx_tail = 0;

void uart_init(uint16_t baud_rate){

    // Set baud rate
    UBRR0H = (baud_rate >> 8);
    UBRR0L = baud_rate;

    // Enable TX, RX, and RX interrupt
    UCSR0B = (1<<TXEN0) | (1<<RXEN0) | (1<<RXCIE0);

    // 8 bit, 1 stop bit, no parity
    UCSR0C = (1<<UCSZ01) | (1<<UCSZ00);

}

void uart_putchar_async(char data){
    uint8_t next = (tx_head + 1) % TX_BUF_SIZE;
    if(next != tx_tail){
        tx_buf[tx_head] = data;
        tx_head = next;
        UCSR0B |= (1<<UDRIE0); // enable data register empty interrupt
    }
}

ISR(USART_UDRE_vect){
    if(tx_tail != tx_head){
        UDR0 = tx_buf[tx_tail];
        tx_tail = (tx_tail + 1) % TX_BUF_SIZE;
    } else {
        UCSR0B &= ~(1<<UDRIE0); // buffer empty, disable interrupt
    }
}

void uart_putchar(char data){
    while(!(UCSR0A & (1<<UDRE0)));
    UDR0 = data;
}

void uart_putstring(char* s){
    while(*s) uart_putchar_async(*s++);
}

void uart_putfloat(float f){
    char buf[16];
    int whole = (int)f;
    int frac = (int)((f - whole) * 100);  // 2 decimal places
    if(frac < 0) frac = -frac;            // handle negative fracs

    int i = 0;
    if(whole < 0){ buf[i++] = '-'; whole = -whole; }
    if(whole == 0){ buf[i++] = '0'; }
    else{
        int start = i;
        while(whole > 0){ buf[i++] = '0' + (whole % 10); whole /= 10; }
        // reverse the digits
        int end = i - 1;
        while(start < end){ char tmp = buf[start]; buf[start++] = buf[end]; buf[end--] = tmp; }
    }
    buf[i++] = '.';
    buf[i++] = '0' + (frac / 10);
    buf[i++] = '0' + (frac % 10);
    buf[i++] = '\r';
    buf[i++] = '\n';
    buf[i] = '\0';

    uart_putstring(buf);
}