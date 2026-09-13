-- proposed_schema.sql

-- =====================================================
-- Schema for Galactic Imperial Space Station (GIS)
-- =====================================================

-- ===========================
-- 1. Personnel Management
-- ===========================

CREATE TABLE species (
    species_id SERIAL PRIMARY KEY,
    species_name VARCHAR(100) UNIQUE NOT NULL,
    description TEXT
);

CREATE TABLE personnel (
    personnel_id SERIAL PRIMARY KEY,
    name VARCHAR(150) NOT NULL,
    rank VARCHAR(50) NOT NULL,
    species_id INT REFERENCES species(species_id),
    section_id INT REFERENCES sections(section_id),
    clearance_level INT NOT NULL,
    status VARCHAR(20) NOT NULL DEFAULT 'Active', -- Active, Inactive, Retired, etc.
    quarters_assignment VARCHAR(50),
    hire_date DATE DEFAULT CURRENT_DATE,
    last_promotion DATE
);

CREATE TABLE roles (
    role_id SERIAL PRIMARY KEY,
    role_name VARCHAR(100) UNIQUE NOT NULL,
    description TEXT
);

CREATE TABLE personnel_roles (
    personnel_id INT REFERENCES personnel(personnel_id),
    role_id INT REFERENCES roles(role_id),
    PRIMARY KEY (personnel_id, role_id)
);

-- ===========================
-- 2. Station Sections
-- ===========================

CREATE TABLE sections (
    section_id SERIAL PRIMARY KEY,
    name VARCHAR(100) UNIQUE NOT NULL,
    level INT NOT NULL, -- Hierarchical level within the station
    security_rating INT NOT NULL, -- 1-10 scale
    capacity INT NOT NULL,
    current_population INT DEFAULT 0,
    description TEXT
);

-- ===========================
-- 3. Life Support Systems
-- ===========================

CREATE TABLE life_support_zones (
    zone_id SERIAL PRIMARY KEY,
    section_id INT REFERENCES sections(section_id),
    oxygen_level DECIMAL(5,2) NOT NULL, -- Percentage
    temperature DECIMAL(4,1) NOT NULL, -- In Celsius
    pressure DECIMAL(6,2) NOT NULL, -- In Pascals
    humidity DECIMAL(4,1) NOT NULL, -- Percentage
    last_checked TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    status VARCHAR(20) NOT NULL DEFAULT 'Operational' -- Operational, Maintenance, Failure
);

-- ===========================
-- 4. Security Management
-- ===========================

CREATE TABLE security_teams (
    team_id SERIAL PRIMARY KEY,
    name VARCHAR(100) UNIQUE NOT NULL,
    leader_id INT REFERENCES personnel(personnel_id),
    shift VARCHAR(50), -- e.g., Day, Night, Rotation
    status VARCHAR(20) NOT NULL DEFAULT 'Active' -- Active, Inactive
);

CREATE TABLE security_events (
    event_id SERIAL PRIMARY KEY,
    event_type VARCHAR(50) NOT NULL, -- e.g., Intrusion, Fire, Medical Emergency
    description TEXT,
    zone_id INT REFERENCES life_support_zones(zone_id),
    reported_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    handled_by INT REFERENCES security_teams(team_id),
    status VARCHAR(20) NOT NULL DEFAULT 'Reported' -- Reported, In Progress, Resolved
);

CREATE TABLE surveillance_cameras (
    camera_id SERIAL PRIMARY KEY,
    zone_id INT REFERENCES life_support_zones(zone_id),
    camera_location VARCHAR(100),
    status VARCHAR(20) NOT NULL DEFAULT 'Operational' -- Operational, Maintenance, Offline
);

-- ===========================
-- 5. Maintenance Management
-- ===========================

CREATE TABLE maintenance_tasks (
    task_id SERIAL PRIMARY KEY,
    system_name VARCHAR(100) NOT NULL, -- e.g., Life Support, Security Systems
    description TEXT NOT NULL,
    priority INT NOT NULL, -- 1-5 scale
    status VARCHAR(20) NOT NULL DEFAULT 'Pending', -- Pending, In Progress, Completed
    assigned_to INT REFERENCES personnel(personnel_id),
    due_date DATE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP
);

CREATE TABLE maintenance_logs (
    log_id SERIAL PRIMARY KEY,
    task_id INT REFERENCES maintenance_tasks(task_id),
    personnel_id INT REFERENCES personnel(personnel_id),
    action_taken TEXT NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ===========================
-- 6. Operations Management
-- ===========================

-- 6.1. Shipping and Logistics

CREATE TABLE cargo_types (
    cargo_type_id SERIAL PRIMARY KEY,
    type_name VARCHAR(100) UNIQUE NOT NULL,
    description TEXT
);

CREATE TABLE shipments (
    shipment_id SERIAL PRIMARY KEY,
    cargo_type_id INT REFERENCES cargo_types(cargo_type_id),
    quantity INT NOT NULL,
    origin_section_id INT REFERENCES sections(section_id),
    destination_section_id INT REFERENCES sections(section_id),
    status VARCHAR(20) NOT NULL DEFAULT 'Scheduled', -- Scheduled, In Transit, Delivered, Delayed
    dispatched_at TIMESTAMP,
    delivered_at TIMESTAMP
);

CREATE TABLE shipping_routes (
    route_id SERIAL PRIMARY KEY,
    origin_section_id INT REFERENCES sections(section_id),
    destination_section_id INT REFERENCES sections(section_id),
    distance DECIMAL(10,2) NOT NULL, -- In light-years or relevant unit
    estimated_travel_time INTERVAL,
    preferred BOOLEAN DEFAULT FALSE
);

-- 6.2. Convoys

CREATE TABLE convoys (
    convoy_id SERIAL PRIMARY KEY,
    name VARCHAR(100) UNIQUE NOT NULL,
    status VARCHAR(20) NOT NULL DEFAULT 'Planned', -- Planned, Active, Completed, Cancelled
    dispatched_at TIMESTAMP,
    expected_arrival TIMESTAMP,
    security_team_id INT REFERENCES security_teams(team_id)
);

CREATE TABLE convoy_shipments (
    convoy_id INT REFERENCES convoys(convoy_id),
    shipment_id INT REFERENCES shipments(shipment_id),
    PRIMARY KEY (convoy_id, shipment_id)
);

-- 6.3. Smuggling Operations

CREATE TABLE smuggling_rings (
    ring_id SERIAL PRIMARY KEY,
    name VARCHAR(100) UNIQUE NOT NULL,
    leader_id INT REFERENCES personnel(personnel_id),
    activity_level INT NOT NULL, -- 1-10 scale
    status VARCHAR(20) NOT NULL DEFAULT 'Active' -- Active, Disbanded, Under Investigation
);

CREATE TABLE smuggled_goods (
    good_id SERIAL PRIMARY KEY,
    ring_id INT REFERENCES smuggling_rings(ring_id),
    cargo_type_id INT REFERENCES cargo_types(cargo_type_id),
    quantity INT NOT NULL,
    smuggled_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ===========================
-- 7. Fleet Management
-- ===========================

CREATE TABLE ship_classes (
    class_id SERIAL PRIMARY KEY,
    class_name VARCHAR(100) UNIQUE NOT NULL,
    description TEXT,
    base_firepower INT,
    base_shield INT,
    base_hull INT,
    speed DECIMAL(5,2) -- In light-years per hour or relevant unit
);

CREATE TABLE ships (
    ship_id SERIAL PRIMARY KEY,
    name VARCHAR(100) UNIQUE NOT NULL,
    class_id INT REFERENCES ship_classes(class_id),
    status VARCHAR(20) NOT NULL DEFAULT 'Docked', -- Docked, In Transit, Under Maintenance, Active
    current_section_id INT REFERENCES sections(section_id),
    assigned_convoy_id INT REFERENCES convoys(convoy_id),
    captain_id INT REFERENCES personnel(personnel_id),
    maintenance_due TIMESTAMP
);

CREATE TABLE ship_upgrades (
    upgrade_id SERIAL PRIMARY KEY,
    ship_id INT REFERENCES ships(ship_id),
    upgrade_name VARCHAR(100) NOT NULL,
    description TEXT,
    upgrade_cost DECIMAL(15,2),
    installed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE ship_crew (
    ship_id INT REFERENCES ships(ship_id),
    personnel_id INT REFERENCES personnel(personnel_id),
    role VARCHAR(50), -- e.g., Pilot, Engineer, Gunner
    PRIMARY KEY (ship_id, personnel_id)
);

-- ===========================
-- 8. Resources Management
-- ===========================

CREATE TABLE resources (
    resource_id SERIAL PRIMARY KEY,
    resource_name VARCHAR(100) UNIQUE NOT NULL,
    description TEXT,
    unit VARCHAR(50) NOT NULL
);

CREATE TABLE resource_stocks (
    resource_id INT REFERENCES resources(resource_id),
    quantity DECIMAL(20,2) NOT NULL,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (resource_id)
);

CREATE TABLE resource_transactions (
    transaction_id SERIAL PRIMARY KEY,
    resource_id INT REFERENCES resources(resource_id),
    transaction_type VARCHAR(20) NOT NULL, -- e.g., Inbound, Outbound, Consumption, Production
    quantity DECIMAL(20,2) NOT NULL,
    related_shipment_id INT REFERENCES shipments(shipment_id),
    related_convoy_id INT REFERENCES convoys(convoy_id),
    related_ship_id INT REFERENCES ships(ship_id),
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    notes TEXT
);

-- ===========================
-- 9. Technology and Advancement
-- ===========================

CREATE TABLE technology_areas (
    area_id SERIAL PRIMARY KEY,
    area_name VARCHAR(100) UNIQUE NOT NULL,
    description TEXT
);

CREATE TABLE technologies (
    tech_id SERIAL PRIMARY KEY,
    tech_name VARCHAR(150) UNIQUE NOT NULL,
    area_id INT REFERENCES technology_areas(area_id),
    description TEXT,
    research_cost DECIMAL(15,2),
    prerequisites TEXT, -- Could be JSON or comma-separated list
    status VARCHAR(20) NOT NULL DEFAULT 'Locked' -- Locked, Researching, Available, Deprecated
);

CREATE TABLE research_projects (
    project_id SERIAL PRIMARY KEY,
    tech_id INT REFERENCES technologies(tech_id),
    lead_researcher_id INT REFERENCES personnel(personnel_id),
    progress DECIMAL(5,2) NOT NULL DEFAULT 0.00, -- Percentage
    start_date DATE DEFAULT CURRENT_DATE,
    expected_completion DATE,
    status VARCHAR(20) NOT NULL DEFAULT 'Ongoing' -- Ongoing, Completed, Failed
);

-- ===========================
-- 10. Galaxy Exploration
-- ===========================

CREATE TABLE star_systems (
    system_id SERIAL PRIMARY KEY,
    system_name VARCHAR(100) UNIQUE NOT NULL,
    coordinates VARCHAR(50) NOT NULL, -- e.g., "X:123, Y:456, Z:789"
    description TEXT,
    populated BOOLEAN DEFAULT FALSE
);

CREATE TABLE planets (
    planet_id SERIAL PRIMARY KEY,
    system_id INT REFERENCES star_systems(system_id),
    planet_name VARCHAR(100) UNIQUE NOT NULL,
    description TEXT,
    habitability_level INT, -- 1-10 scale
    resource_rich BOOLEAN DEFAULT FALSE
);

CREATE TABLE space_anomalies (
    anomaly_id SERIAL PRIMARY KEY,
    system_id INT REFERENCES star_systems(system_id),
    anomaly_type VARCHAR(100) NOT NULL, -- e.g., Black Hole, Nebula
    description TEXT,
    discovery_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE exploration_missions (
    mission_id SERIAL PRIMARY KEY,
    mission_name VARCHAR(100) UNIQUE NOT NULL,
    ship_id INT REFERENCES ships(ship_id),
    destination_system_id INT REFERENCES star_systems(system_id),
    destination_planet_id INT REFERENCES planets(planet_id),
    mission_type VARCHAR(100) NOT NULL, -- e.g., Survey, Resource Extraction, Diplomatic
    status VARCHAR(20) NOT NULL DEFAULT 'Planned', -- Planned, In Progress, Completed, Failed
    start_date TIMESTAMP,
    end_date TIMESTAMP,
    notes TEXT
);

-- ===========================
-- 11. Faction and Diplomacy
-- ===========================

CREATE TABLE factions (
    faction_id SERIAL PRIMARY KEY,
    faction_name VARCHAR(100) UNIQUE NOT NULL,
    description TEXT,
    alignment VARCHAR(50), -- e.g., Ally, Neutral, Enemy
    influence_level INT DEFAULT 0 -- 0-100 scale
);

CREATE TABLE faction_relationships (
    faction1_id INT REFERENCES factions(faction_id),
    faction2_id INT REFERENCES factions(faction_id),
    relationship_status VARCHAR(50) NOT NULL, -- e.g., Ally, Neutral, Hostile
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (faction1_id, faction2_id)
);

CREATE TABLE diplomatic_communications (
    communication_id SERIAL PRIMARY KEY,
    from_faction_id INT REFERENCES factions(faction_id),
    to_faction_id INT REFERENCES factions(faction_id),
    communication_type VARCHAR(50) NOT NULL, -- e.g., Treaty, Declaration, Proposal
    content TEXT NOT NULL,
    sent_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    response_required BOOLEAN DEFAULT FALSE,
    responded BOOLEAN DEFAULT FALSE,
    response_content TEXT,
    response_at TIMESTAMP
);

-- ===========================
-- 12. Inventory Management
-- ===========================

CREATE TABLE inventory_categories (
    category_id SERIAL PRIMARY KEY,
    category_name VARCHAR(100) UNIQUE NOT NULL,
    description TEXT
);

CREATE TABLE inventory_items (
    item_id SERIAL PRIMARY KEY,
    category_id INT REFERENCES inventory_categories(category_id),
    item_name VARCHAR(150) NOT NULL,
    description TEXT,
    quantity INT NOT NULL,
    location VARCHAR(100) NOT NULL, -- e.g., Storage Room, Ship ID
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ===========================
-- 13. Incident and Incident Response
-- ===========================

CREATE TABLE incidents (
    incident_id SERIAL PRIMARY KEY,
    incident_type VARCHAR(100) NOT NULL, -- e.g., Fire, Intrusion, System Failure
    description TEXT NOT NULL,
    reported_by INT REFERENCES personnel(personnel_id),
    reported_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    severity_level INT NOT NULL, -- 1-10 scale
    status VARCHAR(20) NOT NULL DEFAULT 'Reported', -- Reported, In Progress, Resolved, Escalated
    handled_by INT REFERENCES security_teams(team_id),
    resolved_at TIMESTAMP
);

CREATE TABLE incident_logs (
    log_id SERIAL PRIMARY KEY,
    incident_id INT REFERENCES incidents(incident_id),
    action_taken TEXT NOT NULL,
    performed_by INT REFERENCES personnel(personnel_id),
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ===========================
-- 14. Financial Management
-- ===========================

CREATE TABLE accounts (
    account_id SERIAL PRIMARY KEY,
    account_name VARCHAR(100) UNIQUE NOT NULL,
    balance DECIMAL(20,2) NOT NULL DEFAULT 0.00,
    account_type VARCHAR(50) NOT NULL, -- e.g., Operations, Maintenance, Research
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE financial_transactions (
    transaction_id SERIAL PRIMARY KEY,
    account_id INT REFERENCES accounts(account_id),
    transaction_type VARCHAR(20) NOT NULL, -- e.g., Credit, Debit
    amount DECIMAL(20,2) NOT NULL,
    related_shipment_id INT REFERENCES shipments(shipment_id),
    related_convoy_id INT REFERENCES convoys(convoy_id),
    related_project_id INT REFERENCES research_projects(project_id),
    performed_by INT REFERENCES personnel(personnel_id),
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    notes TEXT
);

-- ===========================
-- 15. Research and Development
-- ===========================

CREATE TABLE research_departments (
    department_id SERIAL PRIMARY KEY,
    name VARCHAR(100) UNIQUE NOT NULL,
    head_id INT REFERENCES personnel(personnel_id),
    budget DECIMAL(20,2) NOT NULL DEFAULT 0.00,
    description TEXT
);

CREATE TABLE research_assignments (
    assignment_id SERIAL PRIMARY KEY,
    project_id INT REFERENCES research_projects(project_id),
    personnel_id INT REFERENCES personnel(personnel_id),
    role VARCHAR(50) NOT NULL, -- e.g., Lead Researcher, Assistant
    assigned_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ===========================
-- 16. Health and Medical Services
-- ===========================

CREATE TABLE medical_staff (
    staff_id SERIAL PRIMARY KEY,
    personnel_id INT REFERENCES personnel(personnel_id),
    specialization VARCHAR(100) NOT NULL, -- e.g., Surgeon, Nurse
    license_number VARCHAR(100) UNIQUE,
    status VARCHAR(20) NOT NULL DEFAULT 'Active' -- Active, On Leave, Retired
);

CREATE TABLE medical_records (
    record_id SERIAL PRIMARY KEY,
    personnel_id INT REFERENCES personnel(personnel_id),
    diagnosis TEXT NOT NULL,
    treatment TEXT NOT NULL,
    prescribed_medications TEXT,
    recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    handled_by INT REFERENCES medical_staff(staff_id)
);

CREATE TABLE medical_equipment (
    equipment_id SERIAL PRIMARY KEY,
    equipment_name VARCHAR(100) UNIQUE NOT NULL,
    quantity INT NOT NULL,
    location VARCHAR(100) NOT NULL, -- e.g., Medical Bay, Storage
    status VARCHAR(20) NOT NULL DEFAULT 'Operational' -- Operational, Maintenance, Out of Service
);

-- ===========================
-- 17. Facilities Management
-- ===========================

CREATE TABLE facilities (
    facility_id SERIAL PRIMARY KEY,
    name VARCHAR(100) UNIQUE NOT NULL,
    section_id INT REFERENCES sections(section_id),
    type VARCHAR(50) NOT NULL, -- e.g., Residential, Recreational, Research
    capacity INT NOT NULL,
    current_occupancy INT DEFAULT 0,
    status VARCHAR(20) NOT NULL DEFAULT 'Operational' -- Operational, Under Maintenance, Closed
);

CREATE TABLE facility_reservations (
    reservation_id SERIAL PRIMARY KEY,
    facility_id INT REFERENCES facilities(facility_id),
    personnel_id INT REFERENCES personnel(personnel_id),
    reserved_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    reserved_for TIMESTAMP NOT NULL, -- Date and time of reservation
    duration INTERVAL NOT NULL,
    purpose VARCHAR(100),
    status VARCHAR(20) NOT NULL DEFAULT 'Active' -- Active, Cancelled, Completed
);

-- ===========================
-- 18. Environmental Monitoring
-- ===========================

CREATE TABLE environmental_parameters (
    parameter_id SERIAL PRIMARY KEY,
    zone_id INT REFERENCES life_support_zones(zone_id),
    parameter_name VARCHAR(100) NOT NULL, -- e.g., CO2 Level, Radiation Level
    value DECIMAL(10,2) NOT NULL,
    measured_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    status VARCHAR(20) NOT NULL DEFAULT 'Normal' -- Normal, Warning, Critical
);

CREATE TABLE environmental_alerts (
    alert_id SERIAL PRIMARY KEY,
    parameter_id INT REFERENCES environmental_parameters(parameter_id),
    alert_type VARCHAR(50) NOT NULL, -- e.g., Threshold Breach, System Failure
    description TEXT NOT NULL,
    triggered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    resolved_at TIMESTAMP,
    status VARCHAR(20) NOT NULL DEFAULT 'Triggered' -- Triggered, Acknowledged, Resolved
);

-- ===========================
-- 19. Communication Systems
-- ===========================

CREATE TABLE communication_channels (
    channel_id SERIAL PRIMARY KEY,
    channel_name VARCHAR(100) UNIQUE NOT NULL,
    type VARCHAR(50) NOT NULL, -- e.g., Internal, External, Diplomatic
    status VARCHAR(20) NOT NULL DEFAULT 'Operational' -- Operational, Maintenance, Offline
);

CREATE TABLE communication_logs (
    log_id SERIAL PRIMARY KEY,
    channel_id INT REFERENCES communication_channels(channel_id),
    sender_id INT REFERENCES personnel(personnel_id),
    receiver_id INT REFERENCES personnel(personnel_id),
    message TEXT NOT NULL,
    sent_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    received_at TIMESTAMP
);

-- ===========================
-- 20. Training and Development
-- ===========================

CREATE TABLE training_programs (
    program_id SERIAL PRIMARY KEY,
    program_name VARCHAR(100) UNIQUE NOT NULL,
    description TEXT,
    duration_weeks INT NOT NULL,
    required_clearance_level INT NOT NULL,
    scheduled_start DATE,
    scheduled_end DATE,
    status VARCHAR(20) NOT NULL DEFAULT 'Scheduled' -- Scheduled, Ongoing, Completed, Cancelled
);

CREATE TABLE training_enrollments (
    enrollment_id SERIAL PRIMARY KEY,
    program_id INT REFERENCES training_programs(program_id),
    personnel_id INT REFERENCES personnel(personnel_id),
    enrolled_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completion_status VARCHAR(20) NOT NULL DEFAULT 'Enrolled' -- Enrolled, In Progress, Completed, Failed
);

-- ===========================
-- 21. AI and Automation Systems
-- ===========================

CREATE TABLE ai_modules (
    ai_id SERIAL PRIMARY KEY,
    module_name VARCHAR(100) UNIQUE NOT NULL,
    description TEXT,
    version VARCHAR(50) NOT NULL,
    status VARCHAR(20) NOT NULL DEFAULT 'Active' -- Active, Inactive, Maintenance
);

CREATE TABLE ai_tasks (
    task_id SERIAL PRIMARY KEY,
    ai_id INT REFERENCES ai_modules(ai_id),
    task_description TEXT NOT NULL,
    assigned_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP,
    status VARCHAR(20) NOT NULL DEFAULT 'Assigned' -- Assigned, In Progress, Completed, Failed
);

-- ===========================
-- 22. Event Scheduling
-- ===========================

CREATE TABLE events (
    event_id SERIAL PRIMARY KEY,
    event_name VARCHAR(150) UNIQUE NOT NULL,
    description TEXT,
    scheduled_at TIMESTAMP NOT NULL,
    location VARCHAR(100) NOT NULL, -- e.g., Facility Name, Section
    organizer_id INT REFERENCES personnel(personnel_id),
    status VARCHAR(20) NOT NULL DEFAULT 'Scheduled' -- Scheduled, Ongoing, Completed, Cancelled
);

CREATE TABLE event_attendees (
    event_id INT REFERENCES events(event_id),
    personnel_id INT REFERENCES personnel(personnel_id),
    status VARCHAR(20) NOT NULL DEFAULT 'Registered', -- Registered, Attended, No-show
    PRIMARY KEY (event_id, personnel_id)
);

-- ===========================
-- 23. Emergency Protocols
-- ===========================

CREATE TABLE emergency_protocols (
    protocol_id SERIAL PRIMARY KEY,
    protocol_name VARCHAR(100) UNIQUE NOT NULL,
    description TEXT NOT NULL,
    trigger_conditions TEXT, -- Could be JSON or detailed criteria
    response_steps TEXT, -- Detailed steps for response
    status VARCHAR(20) NOT NULL DEFAULT 'Active' -- Active, Deprecated
);

CREATE TABLE emergency_responses (
    response_id SERIAL PRIMARY KEY,
    protocol_id INT REFERENCES emergency_protocols(protocol_id),
    triggered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    handled_by INT REFERENCES personnel(personnel_id),
    outcome TEXT,
    status VARCHAR(20) NOT NULL DEFAULT 'Initiated' -- Initiated, In Progress, Completed, Failed
);

-- ===========================
-- 24. Data Analytics and Reporting
-- ===========================

CREATE TABLE analytics_reports (
    report_id SERIAL PRIMARY KEY,
    report_name VARCHAR(150) UNIQUE NOT NULL,
    description TEXT,
    generated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    generated_by INT REFERENCES personnel(personnel_id),
    content TEXT -- Could store report data in JSON or other formats
);

CREATE TABLE analytics_queries (
    query_id SERIAL PRIMARY KEY,
    report_id INT REFERENCES analytics_reports(report_id),
    query_text TEXT NOT NULL,
    executed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    executed_by INT REFERENCES personnel(personnel_id),
    status VARCHAR(20) NOT NULL DEFAULT 'Executed' -- Executed, Failed
);

-- ===========================
-- 25. User Authentication and Authorization
-- ===========================

CREATE TABLE user_accounts (
    user_id SERIAL PRIMARY KEY,
    personnel_id INT REFERENCES personnel(personnel_id) UNIQUE,
    username VARCHAR(100) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    salt VARCHAR(255) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_login TIMESTAMP
);

CREATE TABLE user_roles (
    user_id INT REFERENCES user_accounts(user_id),
    role_name VARCHAR(100) NOT NULL,
    PRIMARY KEY (user_id, role_name)
);

CREATE TABLE access_permissions (
    permission_id SERIAL PRIMARY KEY,
    role_name VARCHAR(100) NOT NULL,
    resource VARCHAR(100) NOT NULL, -- e.g., 'ship_management', 'maintenance_tasks'
    permission_level VARCHAR(50) NOT NULL -- e.g., 'read', 'write', 'execute'
);

-- ===========================
-- 26. Auditing and Logging
-- ===========================

CREATE TABLE audit_logs (
    audit_id SERIAL PRIMARY KEY,
    user_id INT REFERENCES user_accounts(user_id),
    action VARCHAR(100) NOT NULL,
    resource VARCHAR(100) NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    details TEXT
);

CREATE TABLE system_logs (
    log_id SERIAL PRIMARY KEY,
    log_type VARCHAR(50) NOT NULL, -- e.g., Error, Warning, Info
    message TEXT NOT NULL,
    logged_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    severity_level INT NOT NULL -- 1-5 scale
);

-- ===========================
-- 27. Advanced Features
-- ===========================

-- 27.1. AI-Driven Predictive Maintenance

CREATE TABLE predictive_maintenance_tasks (
    predictive_task_id SERIAL PRIMARY KEY,
    system_name VARCHAR(100) NOT NULL,
    predicted_issue TEXT NOT NULL,
    confidence_level DECIMAL(5,2) NOT NULL, -- 0.00 to 1.00
    scheduled_date DATE,
    status VARCHAR(20) NOT NULL DEFAULT 'Planned' -- Planned, In Progress, Completed, Cancelled
);

-- 27.2. Interstellar Trade Agreements

CREATE TABLE trade_agreements (
    agreement_id SERIAL PRIMARY KEY,
    partner_faction_id INT REFERENCES factions(faction_id),
    agreement_type VARCHAR(100) NOT NULL, -- e.g., Trade, Non-Aggression, Alliance
    terms TEXT NOT NULL,
    signed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    expiration_date TIMESTAMP,
    status VARCHAR(20) NOT NULL DEFAULT 'Active' -- Active, Expired, Terminated
);

-- 27.3. Resource Allocation Optimization

CREATE TABLE resource_allocation (
    allocation_id SERIAL PRIMARY KEY,
    resource_id INT REFERENCES resources(resource_id),
    allocated_to VARCHAR(100) NOT NULL, -- e.g., 'Research Department', 'Fleet Maintenance'
    quantity DECIMAL(20,2) NOT NULL,
    allocated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    status VARCHAR(20) NOT NULL DEFAULT 'Allocated' -- Allocated, Reallocated, Released
);

-- 27.4. Smuggling Detection Algorithms

CREATE TABLE smuggling_detected (
    detection_id SERIAL PRIMARY KEY,
    convoy_id INT REFERENCES convoys(convoy_id),
    detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    detection_method VARCHAR(100) NOT NULL, -- e.g., AI Analysis, Security Patrol
    confidence_level DECIMAL(5,2) NOT NULL, -- 0.00 to 1.00
    status VARCHAR(20) NOT NULL DEFAULT 'Investigating' -- Investigating, Confirmed, False Positive
);

-- ===========================
-- 28. Backup and Recovery
-- ===========================

-- Note: While backups are typically handled outside the database schema,
-- you can maintain a backup logs table to track backup operations.

CREATE TABLE backup_logs (
    backup_id SERIAL PRIMARY KEY,
    backup_type VARCHAR(50) NOT NULL, -- Full, Incremental, Differential
    performed_by INT REFERENCES personnel(personnel_id),
    backup_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    status VARCHAR(20) NOT NULL DEFAULT 'Completed', -- Completed, Failed
    notes TEXT
);

-- ===========================
-- 29. System Configuration
-- ===========================

CREATE TABLE system_settings (
    setting_id SERIAL PRIMARY KEY,
    setting_name VARCHAR(100) UNIQUE NOT NULL,
    setting_value TEXT NOT NULL,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_by INT REFERENCES personnel(personnel_id)
);

CREATE TABLE feature_flags (
    feature_id SERIAL PRIMARY KEY,
    feature_name VARCHAR(100) UNIQUE NOT NULL,
    is_enabled BOOLEAN DEFAULT FALSE,
    description TEXT,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_by INT REFERENCES personnel(personnel_id)
);

-- ===========================
-- 30. Miscellaneous
-- ===========================

CREATE TABLE notifications (
    notification_id SERIAL PRIMARY KEY,
    personnel_id INT REFERENCES personnel(personnel_id),
    message TEXT NOT NULL,
    read_status BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE documents (
    document_id SERIAL PRIMARY KEY,
    title VARCHAR(150) NOT NULL,
    content TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP,
    uploaded_by INT REFERENCES personnel(personnel_id),
    access_level VARCHAR(50) NOT NULL -- e.g., Public, Restricted, Confidential
);

CREATE TABLE system_metrics (
    metric_id SERIAL PRIMARY KEY,
    metric_name VARCHAR(100) UNIQUE NOT NULL,
    metric_value DECIMAL(20,2) NOT NULL,
    recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- =====================================================
-- End of proposed_schema.sql
-- =====================================================
