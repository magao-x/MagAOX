
// header only version of modbuspp
// 
//
// Moduspp was created by Fanzhe on 5/28/2017.
//
// MagAO-X:
// Cloned from https://github.com/fanzhe98/modbuspp
// On commit 73cabdc, Date:   Sat Nov 24 23:16:51 2018 -0500
//
// License: GPLv3
//
// Header only version created by Jared Males
// -- Added    to all class function definitions, and removed "using namespace std".
// -- commented out all cout diagnostics
//

#ifndef MODBUSPP_MODBUS_H
#define MODBUSPP_MODBUS_H

#include <string>

#include <iostream>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include "modbus_exception.hpp"


#define MAX_MSG_LENGTH 260

///Function Code
enum{
    READ_COILS       = 0x01,
    READ_INPUT_BITS  = 0x02,
    READ_REGS        = 0x03,
    READ_INPUT_REGS  = 0x04,
    WRITE_COIL       = 0x05,
    WRITE_REG        = 0x06,
    WRITE_COILS      = 0x0F,
    WRITE_REGS       = 0x10,
};

///Exception Codes
enum {
    EX_ILLEGAL_FUNCTION = 0x01, // Function Code not Supported
    EX_ILLEGAL_ADDRESS  = 0x02, // Output Address not exists
    EX_ILLEGAL_VALUE    = 0x03, // Output Value not in Range
    EX_SERVER_FAILURE   = 0x04, // Slave Deive Fails to process request
    EX_ACKNOWLEDGE      = 0x05, // Service Need Long Time to Execute
    EX_SERVER_BUSY      = 0x06, // Server Was Unable to Accept MB Request PDU
    EX_GATEWAY_PROBLEMP = 0x0A, // Gateway Path not Available
    EX_GATEWYA_PROBLEMF = 0x0B, // Target Device Failed to Response
};


/// Modbus Operator Class
/**
 * Modbus Operator Class
 * Providing networking support and mobus operation support.
 */
class modbus {
private:
    bool _connected {false};
    uint16_t PORT {502};
    int _socket {0};
    int _msg_id {1};
    int _slaveid {1};
    std::string HOST;
    struct sockaddr_in _server;

    void modbus_build_request(uint8_t *to_send,int address, int func);

    void modbus_read(int address, int amount, int func);
    void modbus_write(int address, int amount, int func, uint16_t *value);

    ssize_t modbus_send(uint8_t *to_send, int length);
    ssize_t modbus_receive(uint8_t *buffer);

    void modbus_error_handle(uint8_t *msg, int func);


public:
    modbus(const std::string & host, uint16_t port);
    explicit modbus(const std::string & host);
    ~modbus();

    bool modbus_connect();
    void modbus_close();

    void modbus_set_slave_id(int id);

    void modbus_read_coils(int address, int amount, bool* buffer);
    void modbus_read_input_bits(int address, int amount, bool* buffer);
    void modbus_read_holding_registers(int address, int amount, uint16_t *buffer);
    void modbus_read_input_registers(int address, int amount, uint16_t *buffer);

    void modbus_write_coil(int address, bool to_write);
    void modbus_write_register(int address, uint16_t value);
    void modbus_write_coils(int address, int amount, bool* value );
    void modbus_write_registers(int address, int amount, uint16_t *value);
};


#if 0 
//-- leaving in for comments until can be normalized.

/**
 * Main Constructor of Modbus Connector Object
 * @param host IP Address of Host
 * @param port Port for the TCP Connection
 * @return     A Modbus Connector Object
 */
  
modbus::modbus( const std::string & host, 
                uint16_t port
              ) : PORT {port}, HOST{host} {}


/**
 * Overloading Modbus Connector Constructor with Default Port Set at 502
 * @param host  IP Address of Host
 * @return      A Modbus Connector Object
 */
  
modbus::modbus(const std::string & host) : HOST{host} {}


/**
 * Destructor of Modbus Connector Object
 */
  
modbus::~modbus(void) {
}


/**
 * Modbus Slave ID Setter
 * @param id  ID of the Modbus Server Slave
 */
  
void modbus::modbus_set_slave_id(int id) {
    _slaveid = id;
}



/**
 * Build up a Modbus/TCP Connection
 * @return   If A Connection Is Successfully Built
 */
  
bool modbus::modbus_connect() {
    if(HOST == "" || PORT == 0) {
        //std::cout << "Missing Host and Port" << std::endl;
        return false;
    } else {
        //std::cout << "Found Proper Host "<< HOST << " and Port " <<PORT <<std::endl;
    }

    _socket = socket(AF_INET, SOCK_STREAM, 0);
    if(_socket == -1) {
        //std::cout <<"Error Opening Socket" <<std::endl;
        return false;
    } else {
        //std::cout <<"Socket Opened Successfully" << std::endl;
    }

    _server.sin_family = AF_INET;
    _server.sin_addr.s_addr = inet_addr(HOST.c_str());
    _server.sin_port = htons(PORT);

    if (connect(_socket, (struct sockaddr*)&_server, sizeof(_server)) < 0) {
        //std::cout<< "Connection Error" << std::endl;
        return false;
    }

    //std::cout<< "Connected" <<std::endl;
    _connected = true;
    return true;
}


/**
 * Close the Modbus/TCP Connection
 */
  
void modbus::modbus_close() {
    close(_socket);
    //std::cout <<"Socket Closed" <<std::endl;
}


/**
 * Modbus Request Builder
 * @param to_send   Message Buffer to Be Sent
 * @param address   Reference Address
 * @param func      Modbus Functional Code
 */
  
void modbus::modbus_build_request(uint8_t *to_send, int address, int func) {
    to_send[0] = (uint8_t) _msg_id >> 8;
    to_send[1] = (uint8_t) (_msg_id & 0x00FF);
    to_send[2] = 0;
    to_send[3] = 0;
    to_send[4] = 0;
    to_send[6] = (uint8_t) _slaveid;
    to_send[7] = (uint8_t) func;
    to_send[8] = (uint8_t) (address >> 8);
    to_send[9] = (uint8_t) (address & 0x00FF);
}


/**
 * Write Request Builder and Sender
 * @param address   Reference Address
 * @param amount    Amount of data to be Written
 * @param func      Modbus Functional Code
 * @param value     Data to Be Written
 */
  
void modbus::modbus_write(int address, int amount, int func, uint16_t *value) {
    if(func == WRITE_COIL || func == WRITE_REG) {
        uint8_t to_send[12];
        modbus_build_request(to_send, address, func);
        to_send[5] = 6;
        to_send[10] = (uint8_t) (value[0] >> 8);
        to_send[11] = (uint8_t) (value[0] & 0x00FF);
        modbus_send(to_send, 12);
    } else if(func == WRITE_REGS){
        uint8_t to_send[13 + 2 * amount];
        modbus_build_request(to_send, address, func);
        to_send[5] = (uint8_t) (5 + 2 * amount);
        to_send[10] = (uint8_t) (amount >> 8);
        to_send[11] = (uint8_t) (amount & 0x00FF);
        to_send[12] = (uint8_t) (2 * amount);
        for(int i = 0; i < amount; i++) {
            to_send[13 + 2 * i] = (uint8_t) (value[i] >> 8);
            to_send[14 + 2 * i] = (uint8_t) (value[i] & 0x00FF);
        }
        modbus_send(to_send, 13 + 2 * amount);
    } else if(func == WRITE_COILS) {
        uint8_t to_send[14 + (amount -1) / 8 ];
        modbus_build_request(to_send, address, func);
        to_send[5] = (uint8_t) (7 + (amount -1) / 8);
        to_send[10] = (uint8_t) (amount >> 8);
        to_send[11] = (uint8_t) (amount >> 8);
        to_send[12] = (uint8_t) ((amount + 7) / 8);
        for(int i = 0; i < amount; i++) {
            to_send[13 + (i - 1) / 8] += (uint8_t) (value[i] << (i % 8));
        }
        modbus_send(to_send, 14 + (amount - 1) / 8);
    }
}


/**
 * Read Request Builder and Sender
 * @param address   Reference Address
 * @param amount    Amount of Data to Read
 * @param func      Modbus Functional Code
 */
  
void modbus::modbus_read(int address, int amount, int func){
    uint8_t to_send[12];
    modbus_build_request(to_send, address, func);
    to_send[5] = 6;
    to_send[10] = (uint8_t) (amount >> 8);
    to_send[11] = (uint8_t) (amount & 0x00FF);
    modbus_send(to_send, 12);
}


/**
 * Read Holding Registers
 * MODBUS FUNCTION 0x03
 * @param address    Reference Address
 * @param amount     Amount of Registers to Read
 * @param buffer     Buffer to Store Data Read from Registers
 */
  
void modbus::modbus_read_holding_registers(int address, int amount, uint16_t *buffer) {
    if(_connected) {
        if(amount > 65535 || address > 65535) {
            throw modbus_amount_exception();
        }
        modbus_read(address, amount, READ_REGS);
        uint8_t to_rec[MAX_MSG_LENGTH];
        modbus_receive(to_rec);
        try {
            modbus_error_handle(to_rec, READ_REGS);
            for(int i = 0; i < amount; i++) {
                buffer[i] = ((uint16_t)to_rec[9 + 2 * i]) << 8;
                buffer[i] += (uint16_t) to_rec[10 + 2 * i];
            }
        } catch (std::exception &e) {
            throw; //e;
        }
    } else {
        throw modbus_connect_exception();
    }
}


/**
 * Read Input Registers 
 * MODBUS FUNCTION 0x04
 * @param address     Reference Address
 * @param amount      Amount of Registers to Read
 * @param buffer      Buffer to Store Data Read from Registers
 */
  
void modbus::modbus_read_input_registers(int address, int amount, uint16_t *buffer) {
    if(_connected){
        if(amount > 65535 || address > 65535) {
            throw modbus_amount_exception();
        }
        modbus_read(address, amount, READ_INPUT_REGS);
        uint8_t to_rec[MAX_MSG_LENGTH];
        modbus_receive(to_rec);
        try {
            modbus_error_handle(to_rec, READ_INPUT_REGS);
            for(int i = 0; i < amount; i++) {
                buffer[i] = ((uint16_t)to_rec[9 + 2 * i]) << 8;
                buffer[i] += (uint16_t) to_rec[10 + 2 * i];
            }
        } catch (std::exception &e) {
            throw; //e;
        }
    } else {
        throw modbus_connect_exception();
    }
}


/**
 * Read Coils           
 * MODBUS FUNCTION 0x01
 * @param address     Reference Address
 * @param amount      Amount of Coils to Read
 * @param buffer      Buffer to Store Data Read from Coils
 */
  
void modbus::modbus_read_coils(int address, int amount, bool *buffer) {
    if(_connected) {
        if(amount > 2040 || address > 65535) {
            throw modbus_amount_exception();
        }
        modbus_read(address, amount, READ_COILS);
        uint8_t to_rec[MAX_MSG_LENGTH];
        modbus_receive(to_rec);
        try {
            modbus_error_handle(to_rec, READ_COILS);
            for(int i = 0; i < amount; i++) {
                buffer[i] = (bool) ((to_rec[9 + i / 8] >> (i % 8)) & 1);
            }
        } catch (std::exception &e) {
            throw; //e;
        }
    } else {
        throw modbus_connect_exception();
    }
}


/**
 * Read Input Bits(Discrete Data)
 * MODBUS FUNCITON 0x02
 * @param address   Reference Address
 * @param amount    Amount of Bits to Read
 * @param buffer    Buffer to store Data Read from Input Bits
 */
  
void modbus::modbus_read_input_bits(int address, int amount, bool* buffer) {
    if(_connected) {
        if(amount > 2040 || address > 65535) {
            throw modbus_amount_exception();
        }
        modbus_read(address, amount, READ_INPUT_BITS);
        uint8_t to_rec[MAX_MSG_LENGTH];
        modbus_receive(to_rec);
        try {
            modbus_error_handle(to_rec, READ_INPUT_BITS);
            for(int i = 0; i < amount; i++) {
                buffer[i] = (bool) ((to_rec[9 + i / 8] >> (i % 8)) & 1);
            }
        } catch (std::exception &e) {
            throw; //e;
        }
    } else {
        throw modbus_connect_exception();
    }
}


/**
 * Write Single Coils
 * MODBUS FUNCTION 0x05
 * @param address    Reference Address
 * @param to_write   Value to be Written to Coil
 */
  
void modbus::modbus_write_coil(int address, bool to_write) {
    if(_connected) {
        if(address > 65535) {
            throw modbus_amount_exception();
        }
        int value = to_write * 0xFF00;
        modbus_write(address, 1, WRITE_COIL, (uint16_t *)&value);
        uint8_t to_rec[MAX_MSG_LENGTH];
        modbus_receive(to_rec);
        try{
            modbus_error_handle(to_rec, WRITE_COIL);
        } catch (std::exception &e) {
            throw; //e;
        }
    } else {
        throw modbus_connect_exception();
    }
}


/**
 * Write Single Register
 * FUCTION 0x06
 * @param address   Reference Address
 * @param value     Value to Be Written to Register
 */
  
void modbus::modbus_write_register(int address, uint16_t value) {
    if(_connected) {
        if(address > 65535) {
            throw modbus_amount_exception();
        }
        modbus_write(address, 1, WRITE_REG, &value);
        uint8_t to_rec[MAX_MSG_LENGTH];
        modbus_receive(to_rec);
        try{
            modbus_error_handle(to_rec, WRITE_COIL);
        } catch (std::exception &e) {
            throw; //e;
        }
    } else {
        throw modbus_connect_exception();
    }
}


/**
 * Write Multiple Coils 
 * MODBUS FUNCTION 0x0F
 * @param address  Reference Address
 * @param amount   Amount of Coils to Write
 * @param value    Values to Be Written to Coils
 */
  
void modbus::modbus_write_coils(int address, int amount, bool *value) {
    if(_connected) {
        if(address > 65535 || amount > 65535) {
            throw modbus_amount_exception();
        }
        uint16_t temp[amount];
        for(int i = 0; i < 4; i++) {
            temp[i] = (uint16_t)value[i];
        }
        modbus_write(address, amount, WRITE_COILS,  temp);
        uint8_t to_rec[MAX_MSG_LENGTH];
        modbus_receive(to_rec);
        try{
            modbus_error_handle(to_rec, WRITE_COILS);
        } catch (std::exception &e) {
            throw; //e;
        }
    } else {
        throw modbus_connect_exception();
    }
}


/**
 * Write Multiple Registers 
 * MODBUS FUNCION 0x10
 * @param address Reference Address
 * @param amount  Amount of Value to Write
 * @param value   Values to Be Written to the Registers
 */
  
void modbus::modbus_write_registers(int address, int amount, uint16_t *value) {
    if(_connected) {
        if(address > 65535 || amount > 65535) {
            throw modbus_amount_exception();
        }
        modbus_write(address, amount, WRITE_REGS, value);
        uint8_t to_rec[MAX_MSG_LENGTH];
        modbus_receive(to_rec);
        try{
            modbus_error_handle(to_rec, WRITE_REGS);
        } catch (std::exception &e) {
            throw; //e;
        }
    } else {
        throw modbus_connect_exception();
    }
}


/**
 * Data Sender
 * @param to_send Request to Be Sent to Server
 * @param length  Length of the Request
 * @return        Size of the request
 */
  
ssize_t modbus::modbus_send(uint8_t *to_send, int length) {
    _msg_id++;
    return send(_socket, to_send, (size_t)length, 0);
}


/**
 * Data Receiver
 * @param buffer Buffer to Store the Data Retrieved
 * @return       Size of Incoming Data
 */
  
ssize_t modbus::modbus_receive(uint8_t *buffer) {
    return recv(_socket, (char *) buffer, MAX_MSG_LENGTH, 0);
}


/**
 * Error Code Handler
 * @param msg   Message Received from the Server
 * @param func  Modbus Functional Code
 */
  
void modbus::modbus_error_handle(uint8_t *msg, int func) {
    if(msg[7] == func + 0x80) {
        switch(msg[8]){
            case EX_ILLEGAL_FUNCTION:
                throw modbus_illegal_function_exception();
            case EX_ILLEGAL_ADDRESS:
                throw modbus_illegal_address_exception();
            case EX_ILLEGAL_VALUE:
                throw modbus_illegal_data_value_exception();
            case EX_SERVER_FAILURE:
                throw modbus_server_failure_exception();
            case EX_ACKNOWLEDGE:
                throw modbus_acknowledge_exception();
            case EX_SERVER_BUSY:
                throw modbus_server_busy_exception();
            case EX_GATEWAY_PROBLEMP:
            case EX_GATEWYA_PROBLEMF:
                throw modbus_gateway_exception();
            default:
                break;
        }
    }
}
#endif

#endif //MODBUSPP_MODBUS_H
