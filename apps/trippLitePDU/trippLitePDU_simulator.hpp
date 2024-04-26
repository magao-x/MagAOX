/** \file trippLitePDU_simulator.hpp
  * \brief The MagAO-X Tripp Lite Power Distribution Unit ontroller Simulator.
  *
  * \ingroup trippLitePDU_files
  */

#ifndef trippLitePDU_simulator_hpp
#define trippLitePDU_simulator_hpp

struct trippLitePDU_simulator
{
    float m_voltage {120};
    float m_frequency {60};
    float m_lowTransferVoltage {70};

    float m_current {4};

    std::vector<int> m_outlets;

    trippLitePDU_simulator()
    {
        m_outlets.resize(8,false);
    }

    int connect( const std::string & ipAddr,
                 const std::string & port 
               )
    {
        static_cast<void>(ipAddr);
        static_cast<void>(port);

        return 0;
    }

    int login( const std::string & user,
               const std::string & pass 
             )
    {
        static_cast<void>(user);
        static_cast<void>(pass);

        return 0;
    }


    void postLogin()
    {
    }

    int turnOutletOn( uint16_t outletNum )
    {
        if(outletNum >= m_outlets.size())
        {
            return -1;
        }

        m_outlets[outletNum] = 1;

        return 0;
    }

    int turnOutletOff( uint16_t outletNum )
    {
        if(outletNum >= m_outlets.size())
        {
            return -1;
        }

        m_outlets[outletNum] = 0;

        return 0;
    }

    int devStatus(std::string & strRead)
    {
        char vstr[64];
        snprintf(vstr, sizeof(vstr), "%0.1f", m_voltage);

        char fstr[64];
        snprintf(fstr, sizeof(fstr), "%0.1f", m_frequency);

        char tvstr[64];
        snprintf(tvstr, sizeof(tvstr), "%0.1f", m_lowTransferVoltage);

        char cstr[64];
        snprintf(cstr, sizeof(cstr), "%0.2f", m_current);

        strRead =  "-------------------------------------------------------------------------------\n";
        strRead += "01: PDUMH20NET2LX 'Device0062'\n";
        strRead += "--------------------------------------------------------------------------------\n";
        strRead += "Device Type:                    PDU\n";
        strRead += "Device Status:                  WARNING        !\n";
        strRead += "\n";
        strRead += "Input Voltage:                  " + std::string(vstr) + " V    \n"     ;
        strRead += "Input Frequency:                " + std::string(fstr) + " Hz       \n"  ;
        strRead += "Low Transfer Voltage:           " + std::string(tvstr) + " V          \n";
        strRead += "\n";
        strRead += "Output Current:                 " + std::string(cstr) + " A - Total  \n";
        strRead += "\n";
        strRead += "Outlets On:                     ";

        //Print outlet numbers of on-outlets, with no space at beginning or end.
        bool prev = false;
        bool none = true;
        for(size_t n=0; n < m_outlets.size(); ++n)
        {
            if(m_outlets[n])
            {
                if(prev) strRead += " ";
                strRead += std::to_string(n+1);
                prev = true;
                none = false;
            }
        }
        if(none)
        {
            strRead += "NONE";
        }

        strRead += "\n";
        
        return 0;
    }
};


#endif //trippLitePDU_simulator_hpp
