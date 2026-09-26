# Hold-out error report

Five disjoint 2,000-entity slices of training Source 1. None of these ids are in the seed-0 200k training sample. Decision rule is the saved matcher: threshold 0.7, cap 11, relative margin 0.75, singleton gate 0.6.

Mean F0.5 0.9451. Min 0.9418. Max 0.9503.

| set | entities | singletons | macro P | macro R | macro F0.5 | TP | FP | FN | blocking_miss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2000 | 112 | 0.9862 | 0.8909 | 0.9503 | 6147 | 77 | 788 | 261 |
| 2 | 2000 | 112 | 0.9848 | 0.8891 | 0.9473 | 6150 | 91 | 780 | 245 |
| 3 | 2000 | 112 | 0.9821 | 0.8799 | 0.9432 | 6098 | 92 | 839 | 275 |
| 4 | 2000 | 112 | 0.9839 | 0.8832 | 0.9430 | 6106 | 97 | 828 | 211 |
| 5 | 2000 | 112 | 0.9866 | 0.8807 | 0.9418 | 6061 | 77 | 850 | 273 |

## Error counts by type

| type | count |
|---|---:|
| false_positive | 384 |
| false_negative | 2820 |
| blocking_miss | 1265 |
| singleton_false_merge | 50 |

## Error counts by type and country

| type | country | count |
|---|---|---:|
| blocking_miss | India | 843 |
| blocking_miss | US | 422 |
| false_negative | India | 1227 |
| false_negative | US | 1593 |
| false_positive | India | 194 |
| false_positive | US | 190 |
| singleton_false_merge | India | 25 |
| singleton_false_merge | US | 25 |

## Examples

### false_positive (15 shown)

- set 1 `S1-625129973` vs `S3-550380926` (India, in_candidates=1)
  S1: Garima Developers (India) Center | Hyderabad, Telangana, Hyderabad, 301-B, Mahaveer Residency, Plot 376-C, Phase 3, Road No 82, Jubilee Hills, Film Nagar
  Other: Garima Developers (India) Center Group | Block C-#360 310-B, Mahaveer Residency, Plot 376-C, Phase 3, Road No 82, Jubilee Hills, Film Nagar, Hyderabad, TG
- set 1 `S1-504756188` vs `S3-796386379` (India, in_candidates=1)
  S1: Kritva Mushroom Private Limited | Plot No. Kh. 22/1, Vill Bamnoli Po Dhulsiras, New Delhi, South West Delhi, Delhi
  Other: Critva Mushroom Private | Pgot No. Kh. 22/1, New Delhi, South West Delhi, DL
- set 1 `S1-707219438` vs `S2-219088039` (India, in_candidates=1)
  S1: Satviki Finance | C/O- Kallol Banerjee., 125/1, Rabindra Nagar, Serampore, Hooghly, West Bengal
  Other: Satviki Finance Group | DOOR NO 016 C/O- KALLOL BANERJEE., 125/1, RABINDRA NAGAR, SERAMPORE, West Bengal
- set 1 `S1-280047886` vs `S3-512568748` (India, in_candidates=1)
  S1: Aardee Memorial Trust | 31, Pawan Vihar, Vaishnav Nagar, Near B.R. Birla Public School, Chopasani, Road, Jodhpur, Rajasthan
  Other: Aardee Memorial Trust Industries | राजस्थान, 40, Jodhpur
- set 1 `S1-81215363` vs `S3-337216879` (US, in_candidates=1)
  S1: Whiteside, Olsen and Fielder | 4420 8th Place, Phoenix, AZ
  Other: Custom | 4420 8th Place, Phoenix, AZ
- set 1 `S1-523253091` vs `S2-780880515` (India, in_candidates=1)
  S1: Real Power | First Floor, H.No.842/A K K Road, Chembumukku, Kakkanad, Ernakulam, Kerala
  Other: Real Power Bakery | NO 8-7 FIRST FLOOR, H.NO.842/A K K ROAD, CHEMBUMUKKU, KAKKANAD, ERNAKULAM, Kerala
- set 1 `S1-879250136` vs `S3-159780896` (India, in_candidates=1)
  S1: Safe Bandhu Private Limited | 79/16, Ahatha Amba Ganj Kesar Ganj Mandi, Meerut, Uttar Pradesh
  Other: Safe Bandhu Private Limited | 
- set 1 `S1-181753147` vs `S2-309704847` (US, in_candidates=1)
  S1: Schnabel Futurewave Inc | 2633 Seevers Avenue, Dallas, TX
  Other: Schnabel Fúturewave Inc | 
- set 1 `S1-60587661` vs `S3-445765972` (India, in_candidates=1)
  S1: Marathwada Consultants | Up Stair, 1St Main, 1St Crs Canara Bank Colony, Ksrtc Layhout Chikkalsandra, 159/2, Bangalore, Karnataka, Bangalore
  Other: Marathwada Consultants Industries Limited | 159/23, Up Stair, 1St Main, 1St Crs Canara Bank Colony, Ksrtc Layhout Chikkalsandra, Bangalore Urban, Bengaluru, KA
- set 1 `S1-775544934` vs `S2-480459813` (India, in_candidates=1)
  S1: Arihant Foods Private Limited | H.No-1918, Sector-3, Part-Hsvp, Rohtak, Haryana
  Other: Arihant Foods Limited Private | #2032/3 SECTOR 3, ROHTAK, Haryana
- set 1 `S1-775544934` vs `S2-507500790` (India, in_candidates=1)
  S1: Arihant Foods Private Limited | H.No-1918, Sector-3, Part-Hsvp, Rohtak, Haryana
  Other: Arihant Foods Private Ltd | H.NO. 2032/3 SECTOR 3, ROHTAK, हरियाणा
- set 1 `S1-775544934` vs `S3-794650328` (India, in_candidates=1)
  S1: Arihant Foods Private Limited | H.No-1918, Sector-3, Part-Hsvp, Rohtak, Haryana
  Other: Arihant Foods Private Limited | #2032 Sector 3, Rohtak, HR
- set 1 `S1-415356868` vs `S3-356539810` (US, in_candidates=1)
  S1: Lambert, Brennan & Foley Laboratories LLC | 348 850, Orem, UT
  Other: Boren Brennan & Foley Laboratories LLC | Utah, Orem, 348-352 850
- set 1 `S1-919027200` vs `S2-537426475` (India, in_candidates=1)
  S1: Taraesh Trading of Shaikpet | 4, 405-411 Jublee Enclave, Hyderabad, Shaikpet, Workflo Hitex Bizness, Telangana
  Other: Taraesha Tmbfdlfeng of Shaikpet | WORKFLO HITEX BIZNESS, 4, 405-411 JUBLEE ENCLAVE, SHAIKPET, Telangana
- set 1 `S1-143345769` vs `S3-618641657` (India, in_candidates=1)
  S1: Subhashree Hospital | House No-32, Block R Inderpuri, New Delhi, Delhi
  Other: Subhashree Solutions | House No-45, New Delhi, DL

### false_negative (15 shown)

- set 1 `S1-235485364` vs `S2-173497338` (India, in_candidates=1)
  S1: Kathiravan Enterprises Corporation | 16/473 Vyjayanthi Appartments Madhavan Nair Road, Near Bhajana Kovil, Calicut, Kozhikode, Kerala
  Other: KATHIRAVAN ENTERPRISES | 
- set 1 `S1-235485364` vs `S3-420169484` (India, in_candidates=1)
  S1: Kathiravan Enterprises Corporation | 16/473 Vyjayanthi Appartments Madhavan Nair Road, Near Bhajana Kovil, Calicut, Kozhikode, Kerala
  Other: Kathiravan Énterprises Corporation | 
- set 1 `S1-454591625` vs `S2-58545865` (India, in_candidates=1)
  S1: Surya Tech Private Limited | Fifth, 506 Satya One Complex, Drive In Road, Ahmadabad City, Ahmedabad, Gujarat
  Other: Limited Surya Tech Partners | H.NO 719 FIFTH, AHMADABAD, AHMEDABAD, Gujarat
- set 1 `S1-174441042` vs `S2-50919296` (India, in_candidates=1)
  S1: Affluence & Brothers Private Limited | B/2 Mahavir Dham, Opp Bmc Hospital, Off Devi Dayal Road, Mulund-West, Mumbai, Mumbai City, Maharashtra
  Other: Affluence & 8róthers Private Limited | B/2/7 MAHAVIR DHAM, OPP BMC HOSPITAL, OFF DEVI DAYAL ROAD, MULUND-WEST, MUMBAI, Maharashtra
- set 1 `S1-625129973` vs `S3-880239134` (India, in_candidates=1)
  S1: Garima Developers (India) Center | Hyderabad, Telangana, Hyderabad, 301-B, Mahaveer Residency, Plot 376-C, Phase 3, Road No 82, Jubilee Hills, Film Nagar
  Other: GDC | 301-B, Hyderabad, Telangana, Hyderabad, Mahaveer Residency, Plot 376-C, Phase 3, Road No 82, Jubilee Hills, Film Nagar
- set 1 `S1-930643517` vs `S2-308606591` (US, in_candidates=1)
  S1: Chambers Health Center | 576 Riverstone Drive, Salem, SC
  Other: Chambers Hea1th Center | 57 RIVERSTONE DR, PO BOX 1649, SALEM, SC
- set 1 `S1-930643517` vs `S2-856000269` (US, in_candidates=1)
  S1: Chambers Health Center | 576 Riverstone Drive, Salem, SC
  Other: CHAMBERS HENCOCEH CENTER | 57 RIVERSTONE DR, SALEM, SC
- set 1 `S1-436743322` vs `S3-865494325` (US, in_candidates=1)
  S1: Viis Inc | 175 Ellebrook Lane, Radcliff, KY
  Other: Inc Vs | 175 Ellebrok Ln, Radcliff, Kentucky
- set 1 `S1-615161329` vs `S3-959510326` (US, in_candidates=1)
  S1: Uptown Coffee | Unit UNIT 1, Salem City, 212 Academy Street, VA
  Other: Uptown Coffee Corporation | 
- set 1 `S1-442740304` vs `S2-278475883` (India, in_candidates=0)
  S1: High Engineering Private Limited | H.No. 227/22, Housing Board Colony, Faridabad, Haryana
  Other: High Engineering Private | 
- set 1 `S1-442740304` vs `S3-888041002` (India, in_candidates=0)
  S1: High Engineering Private Limited | H.No. 227/22, Housing Board Colony, Faridabad, Haryana
  Other: High Engineering Private Limited Enterprises | 
- set 1 `S1-99151563` vs `S2-565705981` (India, in_candidates=1)
  S1: Massey Masala Private Limited | Chaman Complex, 1St Floor, Beside Axis Bank, Sevoke Road, Siliguri, Darjeeling, West Bengal
  Other: Massey Masla Private Limited | CHAMAN COMPLEX, 1ST FLOOR, BESIDE AXIS BANK, SEVOKE ROAD, SILIGURI, West Bengal
- set 1 `S1-286618715` vs `S3-328087007` (US, in_candidates=1)
  S1: Beacon Publishing | 3199 Thompson Avenue, Kingman, AZ
  Other: Beacon Publishing | 
- set 1 `S1-213973805` vs `S3-929239662` (US, in_candidates=1)
  S1: Signature Aviation Inc | 268 Franciscan Drive, Vallejo, CA
  Other: Signature Aviation Incorporated | 
- set 1 `S1-924328537` vs `S3-370380552` (US, in_candidates=1)
  S1: Family Health LLC | 1007 Donoho Street, Clarksville, TX
  Other: Family Health LLC Foundation | Clarksville, Texas, 4007 Donoho St

### blocking_miss (15 shown)

- set 1 `S1-161452717` vs `S3-628587440` (India, in_candidates=0)
  S1: Utech Trading | 27, Shivaji Marg Najafgarh Road, New Delhi, Delhi
  Other: utechtrading.com | 27, New Delhi, दिल्ली
- set 1 `S1-544019441` vs `S2-742004627` (US, in_candidates=0)
  S1: Cardiology Clinic Inc. | 2750 Havenwood Drive, Unit Unit B, City Of Oshkosh, WI
  Other: cclinic.com | 2750 HAVRNWOOD DR, OSHKOSH, WI
- set 1 `S1-454591625` vs `S2-559238308` (India, in_candidates=0)
  S1: Surya Tech Private Limited | Fifth, 506 Satya One Complex, Drive In Road, Ahmadabad City, Ahmedabad, Gujarat
  Other: સૂર્ય ટેક પ્રાઇવેટ લિમિટેડ | AHMEDABAD, H.NO 719 FIFTH, AHMADABAD, Gujarat
- set 1 `S1-454591625` vs `S3-866539847` (India, in_candidates=0)
  S1: Surya Tech Private Limited | Fifth, 506 Satya One Complex, Drive In Road, Ahmadabad City, Ahmedabad, Gujarat
  Other: M/s surya tech private 1imited | 
- set 1 `S1-477271977` vs `S3-521929418` (India, in_candidates=0)
  S1: City Impex Private Limited | H No. 2-9-164, Vikas Nagar, Hanamkonda, Warangal Urban, Telangana
  Other: City Impex Private | Door No 2-9-164, Nalgonda, TG
- set 1 `S1-477271977` vs `S3-970188836` (India, in_candidates=0)
  S1: City Impex Private Limited | H No. 2-9-164, Vikas Nagar, Hanamkonda, Warangal Urban, Telangana
  Other: సిటీ ఇంపెక్స్ ప్రైవేట్ లిమిటెడ్ | TG, Warangal Urban, Hyderabad, H.no 2-9-164
- set 1 `S1-807555702` vs `S2-62463451` (US, in_candidates=0)
  S1: Lucero Oyj Clinic | MO, 216 3rd Street, Desoto
  Other: lucerooyjclinic.com | MO, DE SOTO CP, #216 THIRD STREET
- set 1 `S1-323528295` vs `S2-451943694` (India, in_candidates=0)
  S1: Sun Properties Private Limited | H No.4-3-561/3/A, Opp. Endowment Commissioners Office, Tilak Road, Boggulkunta, Hyderabad, Telangana
  Other: సన్ ప్రాపర్టీస్ ప్రైవేట్ లిమిటెడ్ | H NO.4-3-561/3/A, HYDERABAD, Telangana
- set 1 `S1-286618715` vs `S2-932122659` (US, in_candidates=0)
  S1: Beacon Publishing | 3199 Thompson Avenue, Kingman, AZ
  Other: BEACON SERVICES Enterprises | 
- set 1 `S1-330789535` vs `S3-168952359` (India, in_candidates=0)
  S1: Jai Ventures Private Limited | Plot No - 242, Krishna Kaberi Enclave Ps-Airfield, (Sishu Mandir Road, Hitech), Bhubaneswar, Khordha, Orissa
  Other: ଜୟ ଭେଞ୍ଚର୍ସ୍ ପ୍ରାଇଭେଟ୍ ଲିମିଟେଡ୍ | Plot No - 242, Khordha, OD
- set 1 `S1-330789535` vs `S3-793379939` (India, in_candidates=0)
  S1: Jai Ventures Private Limited | Plot No - 242, Krishna Kaberi Enclave Ps-Airfield, (Sishu Mandir Road, Hitech), Bhubaneswar, Khordha, Orissa
  Other: Jai Ventures Private Limited | 
- set 1 `S1-977173666` vs `S3-766024832` (US, in_candidates=0)
  S1: Brown Anchor Interiors Inc. | 1706 2nd Street, Austin, MN
  Other: Pyragildjax | 2nd St, Austin, Minnesota
- set 1 `S1-515897712` vs `S3-372106628` (US, in_candidates=0)
  S1: Prairie Pioneer North LLC | Broken Arrow, 1709 Maple Avenue, OK
  Other: Prairie Pioneer | 
- set 1 `S1-252736827` vs `S3-50822044` (India, in_candidates=0)
  S1: Red Construction | Flat No 406, Mukund Apartment, Palm Grove Road, Victoria Layout, Bangalore, Karnataka
  Other: Red Conttreuction | 
- set 1 `S1-252736827` vs `S3-598056795` (India, in_candidates=0)
  S1: Red Construction | Flat No 406, Mukund Apartment, Palm Grove Road, Victoria Layout, Bangalore, Karnataka
  Other: ರೆಡ್ ಕನ್‌ಸ್ಟ್ರಕ್ಷನ್ | Flat No 406, Bangalore, KA

### singleton_false_merge (15 shown)

- set 1 `S1-614711574` vs `S2-892156888` (India, in_candidates=1)
  S1: Itl Traders Private Limited | C.A Add:2941/218 Tri Nagar, New Delhi, North Delhi, Delhi
  Other: Itl Traders Limited Private | 
- set 1 `S1-360829167` vs `S2-524793955` (US, in_candidates=1)
  S1: Zaex Inc. | 2025 Fleetwood Drive, Gastonia, NC
  Other: Zaex Inc | 002028 FLEETWOOD DR, GASTONIA, NC
- set 1 `S1-911870605` vs `S3-451728497` (India, in_candidates=1)
  S1: Urban Indian Logistics Private Limited | Ss Plaza Mall, Plot No.-1 Block-A-Mayfield Garden, Sector-47, Gurgaon, Haryana
  Other: अर्बन इंडियन लॉजिस्टिक्स हार्डवेयर प्राइवेट लिमिटेड | Doir No 70 Ss Plaza Mall, Plot No.-1 Block-a-mayfield Garden, Sector-47, Gurgaon, हरियाणा
- set 1 `S1-686606677` vs `S3-154859557` (US, in_candidates=1)
  S1: Denet, L.L.C. | 732 Gurten Street, New Bern, NC
  Other: Denet, LLC | 745 Gurten St, New Bern, North Carolina
- set 1 `S1-39594593` vs `S3-73558424` (US, in_candidates=1)
  S1: TR Gulf Agriculture | 221 Driftwood Lane, Guilford, CT
  Other: XTR Gulf Agriculture | #221 Driftwood Ln, Guilford, Connecticut
- set 1 `S1-546208337` vs `S3-861890293` (US, in_candidates=1)
  S1: Felita's Allied Ventures PLLC | 1922 Foreland Drive, Houston, TX
  Other: ... Harris Allied PLLC Service | 1922 Foreland Drive, Houston, Texas
- set 1 `S1-182537502` vs `S2-808112280` (India, in_candidates=1)
  S1: Yash Seva Samiti | 2B, Plot 18/20 Rebeiro, Building Dhobitalao, Mumbai, Maharashtra
  Other: Yash Seva Samiti Limited | 3B, PLOT 18/20 REBEIRO, BUILDING DHOBITALAO, MUMBAI, Maharashtra
- set 1 `S1-536914889` vs `S2-712800144` (US, in_candidates=1)
  S1: Carter Secure Spa LLC | 746 Belmont Street, Ontario, CA
  Other: A CURE 6 ALL | ONTARIO, 746 BELMONT ST, CA
- set 1 `S1-962744933` vs `S2-377932941` (India, in_candidates=1)
  S1: Delhi India Private Limited | Shop No. 8, Ground Floor, Csc-5, Avantika, Sector-1, Rohini, Delhi, West Delhi, Delhi
  Other: Delhi India Industries Private | SHOP NO. 13, GROUND FLOOR, CSC-5, AVANTIKA, SECTOR-1, ROHINI, DELHI, Delhi
- set 2 `S1-562558122` vs `S3-825264534` (India, in_candidates=1)
  S1: Pratima Consultants | 7/3, Parkland Apartment, Nathan Street Chetpet, Chennai, Tamil Nadu
  Other: pratima consultants industries | #9-741 7/24, Chennai, தமிழ்நாடு
- set 2 `S1-329925128` vs `S3-512059515` (India, in_candidates=1)
  S1: Startup Sharma Center | H No 5-20, Villa No. 46A, Emerald Park, Venkatapuram, Ghatkesar, Rangareddy, Telangana
  Other: Startup Sharma Center Overseas Corp | H No 5-33, Ghatkesar, Rangareddy, TG
- set 2 `S1-477426341` vs `S3-668056491` (India, in_candidates=1)
  S1: EX Online Private Limited | #334, Khatha No 2678/3342Nd Sector Hsr Layout, Bangalore, Karnataka
  Other: EC Online Private Ltd | Bangalore, KA, Bangalore, No 334
- set 2 `S1-826515191` vs `S3-700500254` (US, in_candidates=1)
  S1: 31/70 Hospitality L.L.C. | 328 Saturnia Drive, Georgetown, TX
  Other: 31/70 Hóspitality LLC | 332 Saturnia Dr, Georgetown, Texas
- set 2 `S1-437740294` vs `S2-509058502` (US, in_candidates=1)
  S1: Merlin and Herrera Chesapeake | 5221 Longridge Road, De Witt, NY
  Other: TINSLEY AND HERRERA CHESAPEAKE | 5221 LONGRIDGE RD, JAMESVILLE, NY
- set 2 `S1-424299367` vs `S3-761711741` (US, in_candidates=1)
  S1: Professional Packaging Solutions Corp | 288 Reiman Street, Unit Apartment 1, Sloan, NY
  Other: Professional Packaging Solutions | 
